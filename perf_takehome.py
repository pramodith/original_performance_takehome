"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    """Builder class for constructing kernel programs for the VLIW SIMD machine.

    This class provides methods to build instruction sequences for the custom
    VLIW architecture, including scratch memory allocation, constant handling,
    and kernel construction.

    Attributes:
        instrs: List of instruction bundles that form the program.
        scratch: Dictionary mapping variable names to scratch addresses.
        scratch_debug: Dictionary mapping scratch addresses to (name, length) tuples.
        scratch_ptr: Current pointer into scratch space for allocation.
        const_map: Dictionary mapping constant values to their scratch addresses.
    """

    def __init__(self):
        """Initialize a new KernelBuilder with empty instruction list and scratch space."""
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        """Create a DebugInfo object from the current scratch map.

        Returns:
            DebugInfo: Debug information containing the scratch memory mapping.
        """
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        """Build instruction bundles from a list of engine-slot pairs.

        Args:
            slots: List of (engine, slot) tuples where engine is the execution
                unit and slot contains the operation details.
            vliw: Whether to use VLIW packing (currently unused).

        Returns:
            list: List of instruction dictionaries, each mapping an engine to
                its slots.
        """
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            if engine == "bundle":
                # Pre-packed bundle, pass through directly
                instrs.append(slot)
            else:
                instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        """Add a single instruction slot to the program.

        Args:
            engine: The execution engine name (e.g., "alu", "load", "store", "flow").
            slot: Tuple containing the operation and its operands.
        """
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        """Allocate scratch memory space.

        Args:
            name: Optional name for the scratch variable. If provided, the
                address is stored in the scratch dictionary.
            length: Number of words to allocate. Defaults to 1.

        Returns:
            int: The starting address of the allocated scratch space.

        Raises:
            AssertionError: If allocation would exceed SCRATCH_SIZE.
        """
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        """Get or create a scratch address containing a constant value.

        If the constant already exists in scratch memory, returns its address.
        Otherwise, allocates new scratch space and emits a const load instruction.

        Args:
            val: The constant value to store.
            name: Optional name for the scratch variable.

        Returns:
            int: The scratch address containing the constant.
        """
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        """Build instruction slots for the hash computation.

        Generates ALU instructions implementing the HASH_STAGES operations
        along with debug comparison instructions.

        Args:
            val_hash_addr: Scratch address holding the value to hash (modified in place).
            tmp1: Scratch address for temporary storage.
            tmp2: Scratch address for temporary storage.
            round: Current round number (for debug tracing).
            i: Current batch index (for debug tracing).

        Returns:
            list: List of (engine, slot) tuples implementing the hash stages.
        """
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def add_offset_ops(self, alu_ops, dest, base, offset, tmp_addr):
        """Add ALU op for base + offset, skipping if offset is 0.

        Args:
            alu_ops: List to append ALU operations to.
            dest: List to append resulting addresses to.
            base: Scratch address of base pointer.
            offset: Integer offset to add.
            tmp_addr: Scratch address to use for computed result.
        """
        if offset == 0:
            dest.append(base)
        else:
            offset_const = self.scratch_const(offset)
            alu_ops.append(("+", tmp_addr, base, offset_const))
            dest.append(tmp_addr)

    def _init_memory_vars(self):
        """Load initial variables from memory header into scratch space."""
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        tmp_init = self.alloc_scratch("tmp_init")
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_init, i))
            self.add("load", ("load", self.scratch[v], tmp_init))

    def _alloc_scratch_vectors_for_set(self, num_vectors, set_id):
        """Allocate working vectors for one pipeline set (for double-buffering).

        Args:
            num_vectors: Number of vectors per set
            set_id: Identifier for this set (e.g., 'A' or 'B')

        Returns a dict with allocated scratch vector addresses for this set.
        """
        s = {}
        sfx = f"_{set_id}"

        # Working vectors for idx, val, node_val
        s['tmp_idx'] = [self.alloc_scratch(f"tmp_idx{sfx}_{j}", VLEN) for j in range(num_vectors)]
        s['tmp_val'] = [self.alloc_scratch(f"tmp_val{sfx}_{j}", VLEN) for j in range(num_vectors)]
        s['tmp_node_val'] = [self.alloc_scratch(f"tmp_node_val{sfx}_{j}", VLEN) for j in range(num_vectors)]
        s['tmp_addr'] = [self.alloc_scratch(f"tmp_addr{sfx}_{j}", VLEN) for j in range(num_vectors)]

        # Temporary vectors for computation
        s['tmp1'] = [self.alloc_scratch(f"tmp1{sfx}_{j}", VLEN) for j in range(num_vectors)]
        s['tmp2'] = [self.alloc_scratch(f"tmp2{sfx}_{j}", VLEN) for j in range(num_vectors)]
        s['idx_plus_1'] = [self.alloc_scratch(f"idx_plus_1{sfx}_{j}", VLEN) for j in range(num_vectors)]
        s['idx_plus_2'] = [self.alloc_scratch(f"idx_plus_2{sfx}_{j}", VLEN) for j in range(num_vectors)]

        # Scalar addresses for vload/vstore
        s['vload_addr_idx'] = [self.alloc_scratch(f"vload_addr_idx{sfx}_{j}") for j in range(num_vectors)]
        s['vload_addr_val'] = [self.alloc_scratch(f"vload_addr_val{sfx}_{j}") for j in range(num_vectors)]

        return s

    def _alloc_shared_vectors(self, num_vectors):
        """Allocate shared vectors (constants) used by all pipeline sets."""
        s = {}

        # Hash stage constant vectors
        s['val1_vecs'] = [self.alloc_scratch(f"val1_vec_{hi}", VLEN) for hi in range(len(HASH_STAGES))]
        s['val3_vecs'] = [self.alloc_scratch(f"val3_vec_{hi}", VLEN) for hi in range(len(HASH_STAGES))]

        # Constant vectors for branch computation
        s['two_vec'] = self.alloc_scratch("two_vec", VLEN)
        s['zero_vec'] = self.alloc_scratch("zero_vec", VLEN)
        s['one_vec'] = self.alloc_scratch("one_vec", VLEN)
        s['n_nodes_vec'] = self.alloc_scratch("n_nodes_vec", VLEN)
        s['forest_values_p_vec'] = self.alloc_scratch("forest_values_p_vec", VLEN)

        return s

    def _broadcast_constants(self, body, shared, zero_const, one_const, two_const):
        """Broadcast all constant vectors once before the main loop."""
        # Basic constants
        body.append(("bundle", {"valu": [
            ("vbroadcast", shared['two_vec'], two_const),
            ("vbroadcast", shared['zero_vec'], zero_const),
            ("vbroadcast", shared['one_vec'], one_const),
            ("vbroadcast", shared['n_nodes_vec'], self.scratch["n_nodes"]),
            ("vbroadcast", shared['forest_values_p_vec'], self.scratch["forest_values_p"]),
        ]}))

        # Hash stage constants (split into two bundles, valu limit is 6)
        valu_ops = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES[:3]):
            valu_ops.append(("vbroadcast", shared['val1_vecs'][hi], self.scratch_const(val1)))
            valu_ops.append(("vbroadcast", shared['val3_vecs'][hi], self.scratch_const(val3)))
        body.append(("bundle", {"valu": valu_ops}))

        valu_ops = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES[3:], start=3):
            valu_ops.append(("vbroadcast", shared['val1_vecs'][hi], self.scratch_const(val1)))
            valu_ops.append(("vbroadcast", shared['val3_vecs'][hi], self.scratch_const(val3)))
        body.append(("bundle", {"valu": valu_ops}))

    def _build_gather_round0(self, body, s, shared, num_vectors):
        """Optimized gather for round 0 where all indices are 0.

        Instead of 8 individual loads, we load the root value once and broadcast.
        """
        # Load forest_values[0] (root node) - address is just forest_values_p
        # Use tmp_addr[0] as a temporary scalar location
        body.append(("bundle", {"load": [("load", s['tmp_addr'][0], self.scratch["forest_values_p"])]}))

        # Broadcast root value to all tmp_node_val vectors
        valu_ops = [("vbroadcast", s['tmp_node_val'][v], s['tmp_addr'][0]) for v in range(num_vectors)]
        body.append(("bundle", {"valu": valu_ops}))

        # XOR with tmp_val
        valu_ops = [("^", s['tmp_val'][v], s['tmp_val'][v], s['tmp_node_val'][v]) for v in range(num_vectors)]
        body.append(("bundle", {"valu": valu_ops}))

    def _build_gather_addr_compute(self, body, s, shared, num_vectors):
        """Compute gather addresses: forest_values_p + idx."""
        valu_ops = []
        for v in range(num_vectors):
            valu_ops.append(("+", s['tmp_addr'][v], s['tmp_idx'][v], shared['forest_values_p_vec']))
        body.append(("bundle", {"valu": valu_ops}))

    def _build_gather_loads(self, s, num_vectors):
        """Generate list of load operations for gathering node values.

        Returns list of (load_op, load_op) tuples, one per cycle.
        """
        load_cycles = []
        num_elements = num_vectors * VLEN
        for load_start in range(0, num_elements, SLOT_LIMITS["load"]):
            load_end = min(load_start + SLOT_LIMITS["load"], num_elements)
            load_ops = []
            for li in range(load_start, load_end):
                v, vi = li // VLEN, li % VLEN
                load_ops.append(("load", s['tmp_node_val'][v] + vi, s['tmp_addr'][v] + vi))
            load_cycles.append(load_ops)
        return load_cycles

    def _build_gather_xor(self, body, s, num_vectors):
        """XOR gathered node values with tmp_val."""
        valu_ops = [("^", s['tmp_val'][v], s['tmp_val'][v], s['tmp_node_val'][v]) for v in range(num_vectors)]
        body.append(("bundle", {"valu": valu_ops}))

    def _build_gather_node_values(self, body, s, shared, group_items, num_vectors):
        """Build gather operation to load node values from tree (non-contiguous addresses).

        Overlaps XOR of vector 0 with loading of vector 1 to hide XOR latency.
        XOR of vector 1 is done after all loads complete.
        """
        self._build_gather_addr_compute(body, s, shared, num_vectors)

        # Individual loads in groups of 2 (load slot limit)
        loads_per_vector = VLEN // SLOT_LIMITS["load"]  # 4 cycles to load one vector
        load_cycles = self._build_gather_loads(s, num_vectors)

        for load_cycle, load_ops in enumerate(load_cycles):
            # On cycle 4 (first cycle of loading vector 1), XOR the completed vector 0
            if load_cycle > 0 and load_cycle % loads_per_vector == 0:
                completed_vec = (load_cycle // loads_per_vector) - 1
                valu_ops = [("^", s['tmp_val'][completed_vec], s['tmp_val'][completed_vec], s['tmp_node_val'][completed_vec])]
                body.append(("bundle", {"load": load_ops, "valu": valu_ops}))
            else:
                body.append(("bundle", {"load": load_ops}))

        # XOR last vector
        valu_ops = [("^", s['tmp_val'][num_vectors-1], s['tmp_val'][num_vectors-1], s['tmp_node_val'][num_vectors-1])]
        body.append(("bundle", {"valu": valu_ops}))

    def _build_hash_stages(self, body, s, shared, num_vectors, group_items, round):
        """Build the 6-stage hash computation: val = myhash(val ^ node_val).

        All XORs are done in the gather phase before this function is called.
        """
        # Apply 6 hash stages (constants already broadcast)
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # Fused: tmp1 = val op1 val1, tmp2 = val op3 val3 (independent)
            valu_ops = []
            for v in range(num_vectors):
                valu_ops.append((op1, s['tmp1'][v], s['tmp_val'][v], shared['val1_vecs'][hi]))
                valu_ops.append((op3, s['tmp2'][v], s['tmp_val'][v], shared['val3_vecs'][hi]))
            body.append(("bundle", {"valu": valu_ops}))

            # val = tmp1 op2 tmp2
            valu_ops = []
            for v in range(num_vectors):
                valu_ops.append((op2, s['tmp_val'][v], s['tmp1'][v], s['tmp2'][v]))
            body.append(("bundle", {"valu": valu_ops}))

            # Debug compare after each stage
            for j, i in enumerate(group_items):
                v, vi = j // VLEN, j % VLEN
                body.append(("debug", ("compare", s['tmp_val'][v] + vi, (round, i, "hash_stage", hi))))

    def _build_hash_stages_with_loads(self, body, s, shared, num_vectors, group_items, round, load_ops_list):
        """Build hash stages with interleaved load operations from another group.

        Args:
            load_ops_list: List of load operation lists to interleave (one per available cycle)
        """
        load_idx = 0
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # Fused: tmp1 = val op1 val1, tmp2 = val op3 val3 (independent)
            valu_ops = []
            for v in range(num_vectors):
                valu_ops.append((op1, s['tmp1'][v], s['tmp_val'][v], shared['val1_vecs'][hi]))
                valu_ops.append((op3, s['tmp2'][v], s['tmp_val'][v], shared['val3_vecs'][hi]))

            # Add load ops if available
            if load_idx < len(load_ops_list):
                body.append(("bundle", {"valu": valu_ops, "load": load_ops_list[load_idx]}))
                load_idx += 1
            else:
                body.append(("bundle", {"valu": valu_ops}))

            # val = tmp1 op2 tmp2
            valu_ops = []
            for v in range(num_vectors):
                valu_ops.append((op2, s['tmp_val'][v], s['tmp1'][v], s['tmp2'][v]))

            if load_idx < len(load_ops_list):
                body.append(("bundle", {"valu": valu_ops, "load": load_ops_list[load_idx]}))
                load_idx += 1
            else:
                body.append(("bundle", {"valu": valu_ops}))

            # Debug compare after each stage
            for j, i in enumerate(group_items):
                v, vi = j // VLEN, j % VLEN
                body.append(("debug", ("compare", s['tmp_val'][v] + vi, (round, i, "hash_stage", hi))))

    def _build_branch_computation(self, body, s, shared, num_vectors):
        """Build: idx = 2*idx + (1 if val%2==0 else 2)."""
        # Fused: tmp1 = val % 2, idx_plus_1 = 2*idx+1, idx_plus_2 = 2*idx+2
        valu_ops = []
        for v in range(num_vectors):
            valu_ops.append(("%", s['tmp1'][v], s['tmp_val'][v], shared['two_vec']))
            valu_ops.append(("multiply_add", s['idx_plus_1'][v], s['tmp_idx'][v], shared['two_vec'], shared['one_vec']))
            valu_ops.append(("multiply_add", s['idx_plus_2'][v], s['tmp_idx'][v], shared['two_vec'], shared['two_vec']))
        body.append(("bundle", {"valu": valu_ops}))

        # Select: if even (tmp1=0) pick idx_plus_1, if odd (tmp1=1) pick idx_plus_2
        for v in range(num_vectors):
            body.append(("flow", ("vselect", s['tmp_idx'][v], s['tmp1'][v], s['idx_plus_2'][v], s['idx_plus_1'][v])))

    def _build_branch_computation_with_loads(self, body, s, shared, num_vectors, load_ops_list, load_start_idx):
        """Build branch computation with interleaved loads. Returns next load index."""
        load_idx = load_start_idx

        # Fused: tmp1 = val % 2, idx_plus_1 = 2*idx+1, idx_plus_2 = 2*idx+2
        valu_ops = []
        for v in range(num_vectors):
            valu_ops.append(("%", s['tmp1'][v], s['tmp_val'][v], shared['two_vec']))
            valu_ops.append(("multiply_add", s['idx_plus_1'][v], s['tmp_idx'][v], shared['two_vec'], shared['one_vec']))
            valu_ops.append(("multiply_add", s['idx_plus_2'][v], s['tmp_idx'][v], shared['two_vec'], shared['two_vec']))

        if load_idx < len(load_ops_list):
            body.append(("bundle", {"valu": valu_ops, "load": load_ops_list[load_idx]}))
            load_idx += 1
        else:
            body.append(("bundle", {"valu": valu_ops}))

        # Select: if even (tmp1=0) pick idx_plus_1, if odd (tmp1=1) pick idx_plus_2
        for v in range(num_vectors):
            if load_idx < len(load_ops_list):
                body.append(("bundle", {"flow": [("vselect", s['tmp_idx'][v], s['tmp1'][v], s['idx_plus_2'][v], s['idx_plus_1'][v])], "load": load_ops_list[load_idx]}))
                load_idx += 1
            else:
                body.append(("flow", ("vselect", s['tmp_idx'][v], s['tmp1'][v], s['idx_plus_2'][v], s['idx_plus_1'][v])))

        return load_idx

    def _build_wrap_check(self, body, s, shared, num_vectors):
        """Build: idx = 0 if idx >= n_nodes else idx."""
        # tmp1 = idx < n_nodes
        valu_ops = []
        for v in range(num_vectors):
            valu_ops.append(("<", s['tmp1'][v], s['tmp_idx'][v], shared['n_nodes_vec']))
        body.append(("bundle", {"valu": valu_ops}))

        # idx = select(tmp1, idx, 0)
        for v in range(num_vectors):
            body.append(("flow", ("vselect", s['tmp_idx'][v], s['tmp1'][v], s['tmp_idx'][v], shared['zero_vec'])))

    def _build_wrap_check_with_loads(self, body, s, shared, num_vectors, load_ops_list, load_start_idx):
        """Build wrap check with interleaved loads. Returns next load index."""
        load_idx = load_start_idx

        # tmp1 = idx < n_nodes
        valu_ops = []
        for v in range(num_vectors):
            valu_ops.append(("<", s['tmp1'][v], s['tmp_idx'][v], shared['n_nodes_vec']))

        if load_idx < len(load_ops_list):
            body.append(("bundle", {"valu": valu_ops, "load": load_ops_list[load_idx]}))
            load_idx += 1
        else:
            body.append(("bundle", {"valu": valu_ops}))

        # idx = select(tmp1, idx, 0)
        for v in range(num_vectors):
            if load_idx < len(load_ops_list):
                body.append(("bundle", {"flow": [("vselect", s['tmp_idx'][v], s['tmp1'][v], s['tmp_idx'][v], shared['zero_vec'])], "load": load_ops_list[load_idx]}))
                load_idx += 1
            else:
                body.append(("flow", ("vselect", s['tmp_idx'][v], s['tmp1'][v], s['tmp_idx'][v], shared['zero_vec'])))

        return load_idx

    def _build_debug_compares(self, body, s, group_items, round, tag):
        """Build debug compare instructions for verification."""
        for j, i in enumerate(group_items):
            v, vi = j // VLEN, j % VLEN
            if tag == "idx":
                body.append(("debug", ("compare", s['tmp_idx'][v] + vi, (round, i, tag))))
            elif tag == "val" or tag == "hashed_val":
                body.append(("debug", ("compare", s['tmp_val'][v] + vi, (round, i, tag))))
            elif tag == "node_val":
                body.append(("debug", ("compare", s['tmp_node_val'][v] + vi, (round, i, tag))))
            elif tag == "next_idx" or tag == "wrapped_idx":
                body.append(("debug", ("compare", s['tmp_idx'][v] + vi, (round, i, tag))))

    def _process_single_group(self, body, s, shared, group_items, num_vectors, rounds):
        """Process a single group through all rounds (non-pipelined fallback)."""
        for round in range(rounds):
            self._build_debug_compares(body, s, group_items, round, "idx")
            self._build_debug_compares(body, s, group_items, round, "val")

            if round == 0:
                self._build_gather_round0(body, s, shared, num_vectors)
            else:
                self._build_gather_node_values(body, s, shared, group_items, num_vectors)
            self._build_debug_compares(body, s, group_items, round, "node_val")

            self._build_hash_stages(body, s, shared, num_vectors, group_items, round)
            self._build_debug_compares(body, s, group_items, round, "hashed_val")

            self._build_branch_computation(body, s, shared, num_vectors)
            self._build_debug_compares(body, s, group_items, round, "next_idx")

            self._build_wrap_check(body, s, shared, num_vectors)
            self._build_debug_compares(body, s, group_items, round, "wrapped_idx")

    def _process_group_pair_pipelined(self, body, sA, sB, shared, group_items_A, group_items_B, num_vectors, rounds):
        """Process two groups with pipelined execution - overlap B's gather with A's compute."""
        for round in range(rounds):
            # Debug compares for both groups
            self._build_debug_compares(body, sA, group_items_A, round, "idx")
            self._build_debug_compares(body, sA, group_items_A, round, "val")
            self._build_debug_compares(body, sB, group_items_B, round, "idx")
            self._build_debug_compares(body, sB, group_items_B, round, "val")

            if round == 0:
                # Round 0: use broadcast optimization for both groups
                self._build_gather_round0(body, sA, shared, num_vectors)
                self._build_debug_compares(body, sA, group_items_A, round, "node_val")

                self._build_hash_stages(body, sA, shared, num_vectors, group_items_A, round)
                self._build_debug_compares(body, sA, group_items_A, round, "hashed_val")

                self._build_branch_computation(body, sA, shared, num_vectors)
                self._build_debug_compares(body, sA, group_items_A, round, "next_idx")

                self._build_wrap_check(body, sA, shared, num_vectors)
                self._build_debug_compares(body, sA, group_items_A, round, "wrapped_idx")

                # Same for B
                self._build_gather_round0(body, sB, shared, num_vectors)
                self._build_debug_compares(body, sB, group_items_B, round, "node_val")

                self._build_hash_stages(body, sB, shared, num_vectors, group_items_B, round)
                self._build_debug_compares(body, sB, group_items_B, round, "hashed_val")

                self._build_branch_computation(body, sB, shared, num_vectors)
                self._build_debug_compares(body, sB, group_items_B, round, "next_idx")

                self._build_wrap_check(body, sB, shared, num_vectors)
                self._build_debug_compares(body, sB, group_items_B, round, "wrapped_idx")
            else:
                # Rounds 1+: Pipeline A's compute with B's gather loads

                # A: address compute
                self._build_gather_addr_compute(body, sA, shared, num_vectors)

                # A: gather loads (no overlap yet)
                load_cycles_A = self._build_gather_loads(sA, num_vectors)
                for load_ops in load_cycles_A:
                    body.append(("bundle", {"load": load_ops}))

                # A: XOR
                self._build_gather_xor(body, sA, num_vectors)
                self._build_debug_compares(body, sA, group_items_A, round, "node_val")

                # B: address compute
                self._build_gather_addr_compute(body, sB, shared, num_vectors)

                # Generate B's gather loads to interleave with A's hash
                load_cycles_B = self._build_gather_loads(sB, num_vectors)

                # A: hash stages with B's gather loads interleaved
                self._build_hash_stages_with_loads(body, sA, shared, num_vectors, group_items_A, round, load_cycles_B)
                self._build_debug_compares(body, sA, group_items_A, round, "hashed_val")

                # A: branch and wrap (B's loads should be done by now)
                self._build_branch_computation(body, sA, shared, num_vectors)
                self._build_debug_compares(body, sA, group_items_A, round, "next_idx")

                self._build_wrap_check(body, sA, shared, num_vectors)
                self._build_debug_compares(body, sA, group_items_A, round, "wrapped_idx")

                # B: XOR (loads completed during A's hash)
                self._build_gather_xor(body, sB, num_vectors)
                self._build_debug_compares(body, sB, group_items_B, round, "node_val")

                # B: hash, branch, wrap (no interleaving for now)
                self._build_hash_stages(body, sB, shared, num_vectors, group_items_B, round)
                self._build_debug_compares(body, sB, group_items_B, round, "hashed_val")

                self._build_branch_computation(body, sB, shared, num_vectors)
                self._build_debug_compares(body, sB, group_items_B, round, "next_idx")

                self._build_wrap_check(body, sB, shared, num_vectors)
                self._build_debug_compares(body, sB, group_items_B, round, "wrapped_idx")

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """Build the complete kernel program for tree traversal."""
        # Initialize memory variables
        self._init_memory_vars()

        # Allocate scalar constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause for debug synchronization
        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        body = []

        # Setup: allocate vectors and broadcast constants
        bundle_size = SLOT_LIMITS["load"] * VLEN  # 16 elements per cycle
        num_vectors = SLOT_LIMITS["load"]  # 2 vectors

        # Allocate two sets of working vectors for pipelining
        sA = self._alloc_scratch_vectors_for_set(num_vectors, 'A')
        sB = self._alloc_scratch_vectors_for_set(num_vectors, 'B')
        shared = self._alloc_shared_vectors(num_vectors)

        self._broadcast_constants(body, shared, zero_const, one_const, two_const)

        # Process groups in pairs for pipelining
        group_starts = list(range(0, batch_size, bundle_size))
        num_groups = len(group_starts)

        for pair_idx in range(0, num_groups, 2):
            group_idx_A = pair_idx
            group_idx_B = pair_idx + 1 if pair_idx + 1 < num_groups else None

            group_start_A = group_starts[group_idx_A]
            group_items_A = list(range(group_start_A, min(group_start_A + bundle_size, batch_size)))

            # Compute addresses for group A
            alu_ops = []
            for v in range(num_vectors):
                offset = group_start_A + v * VLEN
                offset_const = self.scratch_const(offset)
                alu_ops.append(("+", sA['vload_addr_idx'][v], self.scratch["inp_indices_p"], offset_const))
                alu_ops.append(("+", sA['vload_addr_val'][v], self.scratch["inp_values_p"], offset_const))
            body.append(("bundle", {"alu": alu_ops}))

            # Load idx and val for group A
            body.append(("bundle", {"load": [("vload", sA['tmp_idx'][v], sA['vload_addr_idx'][v]) for v in range(num_vectors)]}))
            body.append(("bundle", {"load": [("vload", sA['tmp_val'][v], sA['vload_addr_val'][v]) for v in range(num_vectors)]}))

            if group_idx_B is not None:
                # We have a pair - use pipelined processing
                group_start_B = group_starts[group_idx_B]
                group_items_B = list(range(group_start_B, min(group_start_B + bundle_size, batch_size)))

                # Compute addresses for group B
                alu_ops = []
                for v in range(num_vectors):
                    offset = group_start_B + v * VLEN
                    offset_const = self.scratch_const(offset)
                    alu_ops.append(("+", sB['vload_addr_idx'][v], self.scratch["inp_indices_p"], offset_const))
                    alu_ops.append(("+", sB['vload_addr_val'][v], self.scratch["inp_values_p"], offset_const))
                body.append(("bundle", {"alu": alu_ops}))

                # Load idx and val for group B
                body.append(("bundle", {"load": [("vload", sB['tmp_idx'][v], sB['vload_addr_idx'][v]) for v in range(num_vectors)]}))
                body.append(("bundle", {"load": [("vload", sB['tmp_val'][v], sB['vload_addr_val'][v]) for v in range(num_vectors)]}))

                # Process both groups with pipelining
                self._process_group_pair_pipelined(body, sA, sB, shared, group_items_A, group_items_B, num_vectors, rounds)

                # Store results for both groups
                body.append(("bundle", {"store": [("vstore", sA['vload_addr_idx'][v], sA['tmp_idx'][v]) for v in range(num_vectors)]}))
                body.append(("bundle", {"store": [("vstore", sA['vload_addr_val'][v], sA['tmp_val'][v]) for v in range(num_vectors)]}))
                body.append(("bundle", {"store": [("vstore", sB['vload_addr_idx'][v], sB['tmp_idx'][v]) for v in range(num_vectors)]}))
                body.append(("bundle", {"store": [("vstore", sB['vload_addr_val'][v], sB['tmp_val'][v]) for v in range(num_vectors)]}))
            else:
                # Odd group at the end - process single group
                self._process_single_group(body, sA, shared, group_items_A, num_vectors, rounds)

                # Store results
                body.append(("bundle", {"store": [("vstore", sA['vload_addr_idx'][v], sA['tmp_idx'][v]) for v in range(num_vectors)]}))
                body.append(("bundle", {"store": [("vstore", sA['vload_addr_val'][v], sA['tmp_val'][v]) for v in range(num_vectors)]}))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    """Execute a kernel test with the specified parameters.

    Builds and runs a kernel on a randomly generated tree, comparing results
    against the reference implementation.

    Args:
        forest_height: Height of the binary tree to generate.
        rounds: Number of traversal rounds to perform.
        batch_size: Number of parallel traversals in each batch.
        seed: Random seed for reproducible tree generation. Defaults to 123.
        trace: Whether to enable execution tracing. Defaults to False.
        prints: Whether to print intermediate values. Defaults to False.

    Returns:
        int: The number of cycles taken to execute the kernel.

    Raises:
        AssertionError: If the kernel output does not match the reference.
    """
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    """Unit tests for the kernel implementation and reference kernels."""

    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        """Test kernel execution with tracing enabled.

        Runs a full-scale example for performance testing with trace output.
        """
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        """Test kernel execution and report cycle count.

        Runs the kernel with standard parameters and prints cycle count
        and speedup over baseline.
        """
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
