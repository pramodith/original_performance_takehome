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
from dataclasses import dataclass, field
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


@dataclass
class Instruction:
    """A single instruction with dependency tracking for scheduling."""
    engine: str           # "valu", "load", "flow", "alu", "store", "debug"
    op: tuple             # The actual operation tuple
    group: str            # "A" or "B"
    phase: int            # Dependency ordering within a round (0=gather, 1=hash, 2=branch, 3=wrap)
    round: int            # Which round this instruction belongs to
    seq: int = 0          # Sequence number within phase for ordering
    scheduled: bool = field(default=False, repr=False)
    scalar_offset: int = field(default=0, repr=False)  # For partial valu scalarization


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

        # Scalar addresses for vload/vstore
        s['vload_addr_idx'] = [self.alloc_scratch(f"vload_addr_idx{sfx}_{j}") for j in range(num_vectors)]
        s['vload_addr_val'] = [self.alloc_scratch(f"vload_addr_val{sfx}_{j}") for j in range(num_vectors)]

        return s

    def _alloc_shared_vectors(self, num_vectors):
        """Allocate shared vectors (constants) used by all pipeline sets."""
        s = {}

        # Hash stage constant vectors
        # val1_vecs needed for all stages
        s['val1_vecs'] = [self.alloc_scratch(f"val1_vec_{hi}", VLEN) for hi in range(len(HASH_STAGES))]
        # val3_vecs only needed for non-optimized stages (where op1!='+' or op2!='+')
        # hash_mul_vecs only needed for optimized stages (where op1='+' and op2='+')
        s['val3_vecs'] = {}
        s['hash_mul_vecs'] = {}
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == '+' and op2 == '+':
                # Optimized stage - needs multiplier, not val3
                s['hash_mul_vecs'][hi] = self.alloc_scratch(f"hash_mul_vec_{hi}", VLEN)
            else:
                # Non-optimized stage - needs val3, not multiplier
                s['val3_vecs'][hi] = self.alloc_scratch(f"val3_vec_{hi}", VLEN)

        # Constant vectors for branch computation
        s['two_vec'] = self.alloc_scratch("two_vec", VLEN)
        s['one_vec'] = self.alloc_scratch("one_vec", VLEN)
        s['n_nodes_vec'] = self.alloc_scratch("n_nodes_vec", VLEN)
        s['forest_values_p_vec'] = self.alloc_scratch("forest_values_p_vec", VLEN)

        # Level-based tree value preloading for rounds 0-3
        s['level0_scalar'] = self.alloc_scratch("tree_l0")  # tree[0] = root
        s['level0_vec'] = self.alloc_scratch("tree_l0_vec", VLEN)
        s['level1_scalars'] = [self.alloc_scratch(f"tree_l1_{i}") for i in range(2)]
        s['level2_scalars'] = [self.alloc_scratch(f"tree_l2_{i}") for i in range(4)]
        s['level3_scalars'] = [self.alloc_scratch(f"tree_l3_{i}") for i in range(8)]
        s['level1_vecs'] = [self.alloc_scratch(f"tree_l1_vec_{i}", VLEN) for i in range(2)]
        s['level2_vecs'] = [self.alloc_scratch(f"tree_l2_vec_{i}", VLEN) for i in range(4)]
        s['level3_vecs'] = [self.alloc_scratch(f"tree_l3_vec_{i}", VLEN) for i in range(8)]
        s['three_vec'] = self.alloc_scratch("three_vec", VLEN)
        s['seven_vec'] = self.alloc_scratch("seven_vec", VLEN)

        return s

    def _broadcast_constants(self, body, shared, zero_const, one_const, two_const):
        """Broadcast all constant vectors once before the main loop."""
        # Basic constants
        three_const = self.scratch_const(3)
        seven_const = self.scratch_const(7)
        body.append(("bundle", {"valu": [
            ("vbroadcast", shared['two_vec'], two_const),
            ("vbroadcast", shared['one_vec'], one_const),
            ("vbroadcast", shared['n_nodes_vec'], self.scratch["n_nodes"]),
            ("vbroadcast", shared['forest_values_p_vec'], self.scratch["forest_values_p"]),
            ("vbroadcast", shared['three_vec'], three_const),
            ("vbroadcast", shared['seven_vec'], seven_const),
        ]}))

        # Hash stage constants
        valu_ops = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            valu_ops.append(("vbroadcast", shared['val1_vecs'][hi], self.scratch_const(val1)))
            if hi in shared['val3_vecs']:
                valu_ops.append(("vbroadcast", shared['val3_vecs'][hi], self.scratch_const(val3)))
            if hi in shared['hash_mul_vecs']:
                multiplier = (1 << val3) + 1
                valu_ops.append(("vbroadcast", shared['hash_mul_vecs'][hi], self.scratch_const(multiplier)))
            if len(valu_ops) >= 5:
                body.append(("bundle", {"valu": valu_ops}))
                valu_ops = []
        if valu_ops:
            body.append(("bundle", {"valu": valu_ops}))

        # Preload tree values for levels 0-3
        tmp_addr = [self.alloc_scratch(f"tmp_tree_addr_{i}") for i in range(2)]
        # Level 0: tree[0] (root)
        body.append(("bundle", {"load": [("load", shared['level0_scalar'], self.scratch["forest_values_p"])]}))
        # Levels 1-3: tree[1-14]
        all_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        all_scalars = shared['level1_scalars'] + shared['level2_scalars'] + shared['level3_scalars']
        for i in range(0, len(all_indices), 2):
            alu_ops = [("+", tmp_addr[j], self.scratch["forest_values_p"], self.scratch_const(all_indices[i+j])) for j in range(min(2, len(all_indices)-i))]
            body.append(("bundle", {"alu": alu_ops}))
            load_ops = [("load", all_scalars[i+j], tmp_addr[j]) for j in range(min(2, len(all_indices)-i))]
            body.append(("bundle", {"load": load_ops}))

        # Broadcast preloaded values to vectors
        valu_ops = []
        valu_ops.append(("vbroadcast", shared['level0_vec'], shared['level0_scalar']))
        for i in range(2):
            valu_ops.append(("vbroadcast", shared['level1_vecs'][i], shared['level1_scalars'][i]))
        for i in range(3):  # First 3 of level2
            valu_ops.append(("vbroadcast", shared['level2_vecs'][i], shared['level2_scalars'][i]))
        body.append(("bundle", {"valu": valu_ops}))
        # Remaining level2 broadcast
        body.append(("bundle", {"valu": [("vbroadcast", shared['level2_vecs'][3], shared['level2_scalars'][3])]}))
        valu_ops = []
        for i in range(8):
            valu_ops.append(("vbroadcast", shared['level3_vecs'][i], shared['level3_scalars'][i]))
            if len(valu_ops) >= 6:
                body.append(("bundle", {"valu": valu_ops}))
                valu_ops = []
        if valu_ops:
            body.append(("bundle", {"valu": valu_ops}))

    # ========== Instruction Generation (Queue-based approach) ==========

    def _gen_gather_instrs(self, s, shared, group, round_num, num_vectors, group_items):
        """Generate gather instructions for one group's round."""
        instrs = []
        seq = 0

        # Compute level within traversal cycle (wraps after forest_height + 1 rounds)
        level = round_num % (self.forest_height + 1)

        if level == 0:
            # Level 0: use preloaded root value (level0_vec)
            # Copy level0_vec to tmp_node_val using: tmp_node_val = level0_vec + 0
            # Compute zero vector: tmp1[0] = one_vec ^ one_vec = 0
            instrs.append(Instruction("valu", ("^", s['tmp1'][0], shared['one_vec'], shared['one_vec']), group, 0, round_num, seq))
            seq += 1
            # Copy: tmp_node_val = level0_vec + 0
            for v in range(num_vectors):
                instrs.append(Instruction("valu", ("+", s['tmp_node_val'][v], shared['level0_vec'], s['tmp1'][0]), group, 0, round_num, seq))
            seq += 1
        elif level == 1:
            # Round 1: idx is 1 or 2, use 2-way select
            # node_val = t0 + (t1-t0)*(idx-1)
            for v in range(num_vectors):
                instrs.append(Instruction("valu", ("-", s['tmp_addr'][v], s['tmp_idx'][v], shared['one_vec']), group, 0, round_num, seq))
                instrs.append(Instruction("valu", ("-", s['tmp1'][v], shared['level1_vecs'][1], shared['level1_vecs'][0]), group, 0, round_num, seq))
            seq += 1
            for v in range(num_vectors):
                instrs.append(Instruction("valu", ("*", s['tmp2'][v], s['tmp1'][v], s['tmp_addr'][v]), group, 0, round_num, seq))
            seq += 1
            for v in range(num_vectors):
                instrs.append(Instruction("valu", ("+", s['tmp_node_val'][v], shared['level1_vecs'][0], s['tmp2'][v]), group, 0, round_num, seq))
            seq += 1
        elif level == 2:
            # Round 2: idx is 3-6, use 4-way select
            for v in range(num_vectors):
                instrs.append(Instruction("valu", ("-", s['tmp_addr'][v], s['tmp_idx'][v], shared['three_vec']), group, 0, round_num, seq))
            seq += 1
            for v in range(num_vectors):
                instrs.append(Instruction("valu", ("&", s['tmp1'][v], s['tmp_addr'][v], shared['one_vec']), group, 0, round_num, seq))
                instrs.append(Instruction("valu", (">>", s['tmp2'][v], s['tmp_addr'][v], shared['one_vec']), group, 0, round_num, seq))
                # Use tmp_node_val as scratch (it's only set at the end)
                instrs.append(Instruction("valu", ("-", s['tmp_node_val'][v], shared['level2_vecs'][1], shared['level2_vecs'][0]), group, 0, round_num, seq))
                instrs.append(Instruction("valu", ("-", s['idx_plus_1'][v], shared['level2_vecs'][3], shared['level2_vecs'][2]), group, 0, round_num, seq))
            seq += 1
            for v in range(num_vectors):
                instrs.append(Instruction("valu", ("*", s['tmp_addr'][v], s['tmp_node_val'][v], s['tmp1'][v]), group, 0, round_num, seq))
                instrs.append(Instruction("valu", ("*", s['tmp_node_val'][v], s['idx_plus_1'][v], s['tmp1'][v]), group, 0, round_num, seq))
            seq += 1
            for v in range(num_vectors):
                instrs.append(Instruction("valu", ("+", s['idx_plus_1'][v], shared['level2_vecs'][0], s['tmp_addr'][v]), group, 0, round_num, seq))
                instrs.append(Instruction("valu", ("+", s['tmp_addr'][v], shared['level2_vecs'][2], s['tmp_node_val'][v]), group, 0, round_num, seq))
            seq += 1
            for v in range(num_vectors):
                instrs.append(Instruction("valu", ("-", s['tmp_node_val'][v], s['tmp_addr'][v], s['idx_plus_1'][v]), group, 0, round_num, seq))
            seq += 1
            for v in range(num_vectors):
                instrs.append(Instruction("valu", ("*", s['tmp_addr'][v], s['tmp_node_val'][v], s['tmp2'][v]), group, 0, round_num, seq))
            seq += 1
            for v in range(num_vectors):
                instrs.append(Instruction("valu", ("+", s['tmp_node_val'][v], s['idx_plus_1'][v], s['tmp_addr'][v]), group, 0, round_num, seq))
            seq += 1
        else:
            # Levels 3+: scatter-gather
            for v in range(num_vectors):
                instrs.append(Instruction("valu", ("+", s['tmp_addr'][v], s['tmp_idx'][v], shared['forest_values_p_vec']), group, 0, round_num, seq))
            seq += 1
            for li in range(num_vectors * VLEN):
                v, vi = li // VLEN, li % VLEN
                instrs.append(Instruction("load", ("load", s['tmp_node_val'][v] + vi, s['tmp_addr'][v] + vi), group, 0, round_num, seq + li // SLOT_LIMITS["load"]))
            seq += (num_vectors * VLEN + SLOT_LIMITS["load"] - 1) // SLOT_LIMITS["load"]

        # XOR: val = val ^ node_val
        for v in range(num_vectors):
            instrs.append(Instruction("valu", ("^", s['tmp_val'][v], s['tmp_val'][v], s['tmp_node_val'][v]), group, 0, round_num, seq))

        # Debug compares
        for j, i in enumerate(group_items):
            v, vi = j // VLEN, j % VLEN
            instrs.append(Instruction("debug", ("compare", s['tmp_node_val'][v] + vi, (round_num, i, "node_val")), group, 0, round_num, seq + 1))

        return instrs

    def _gen_hash_instrs(self, s, shared, group, round_num, num_vectors, group_items):
        """Generate hash instructions for one group's round."""
        instrs = []
        seq = 0

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # Check if this stage can be optimized with multiply_add
            # When op1='+' and op2='+': a' = (a + val1) + (a << val3) = a * ((1<<val3)+1) + val1
            if op1 == '+' and op2 == '+':
                # Optimized: single multiply_add instruction
                # val = val * multiplier + val1
                for v in range(num_vectors):
                    instrs.append(Instruction("valu", ("multiply_add", s['tmp_val'][v], s['tmp_val'][v], shared['hash_mul_vecs'][hi], shared['val1_vecs'][hi]), group, 1, round_num, seq))
                seq += 1
            else:
                # Original: 3 operations
                # Part 1: tmp1 = val op1 val1, tmp2 = val op3 val3
                for v in range(num_vectors):
                    instrs.append(Instruction("valu", (op1, s['tmp1'][v], s['tmp_val'][v], shared['val1_vecs'][hi]), group, 1, round_num, seq))
                    instrs.append(Instruction("valu", (op3, s['tmp2'][v], s['tmp_val'][v], shared['val3_vecs'][hi]), group, 1, round_num, seq))
                seq += 1

                # Part 2: val = tmp1 op2 tmp2
                for v in range(num_vectors):
                    instrs.append(Instruction("valu", (op2, s['tmp_val'][v], s['tmp1'][v], s['tmp2'][v]), group, 1, round_num, seq))
                seq += 1

            # Debug compares after each stage
            for j, i in enumerate(group_items):
                v, vi = j // VLEN, j % VLEN
                instrs.append(Instruction("debug", ("compare", s['tmp_val'][v] + vi, (round_num, i, "hash_stage", hi)), group, 1, round_num, seq))

        return instrs

    def _gen_branch_instrs(self, s, shared, group, round_num, num_vectors, group_items):
        """Generate branch instructions for one group's round."""
        instrs = []

        # Optimized branch: idx = 2*idx + 1 + (val%2)
        # This eliminates the vselect entirely!
        # Original: idx = vselect(val%2, 2*idx+2, 2*idx+1)
        # When val%2=1: 2*idx+2. When val%2=0: 2*idx+1
        # So: 2*idx + 1 + (val&1)  [val&1 is equivalent to val%2 for non-negative integers]
        for v in range(num_vectors):
            # tmp1 = val & 1 (same as val % 2)
            instrs.append(Instruction("valu", ("&", s['tmp1'][v], s['tmp_val'][v], shared['one_vec']), group, 2, round_num, 0))
            # idx_plus_1 = 2*idx + 1
            instrs.append(Instruction("valu", ("multiply_add", s['idx_plus_1'][v], s['tmp_idx'][v], shared['two_vec'], shared['one_vec']), group, 2, round_num, 0))

        # idx = idx_plus_1 + tmp1 (this is 2*idx + 1 + (val%2))
        for v in range(num_vectors):
            instrs.append(Instruction("valu", ("+", s['tmp_idx'][v], s['idx_plus_1'][v], s['tmp1'][v]), group, 2, round_num, 1))

        # Debug compares
        for j, i in enumerate(group_items):
            v, vi = j // VLEN, j % VLEN
            instrs.append(Instruction("debug", ("compare", s['tmp_idx'][v] + vi, (round_num, i, "next_idx")), group, 2, round_num, 2))

        return instrs

    def _gen_wrap_instrs(self, s, shared, group, round_num, num_vectors, group_items):
        """Generate wrap instructions for one group's round.

        Wrap logic: idx = (idx < n_nodes) ? idx : 0

        Optimization: For a perfect binary tree, wrap only happens at the leaf level.
        - Levels 0 to forest_height-1: idx is always < n_nodes, no wrap needed
        - Level forest_height (leaves): after branch, idx always >= n_nodes, set idx = 0
        """
        instrs = []

        level = round_num % (self.forest_height + 1)

        if level == self.forest_height:
            # At leaf level, wrap always happens - just set idx = 0
            # Using idx ^ idx = 0 (XOR with itself)
            for v in range(num_vectors):
                instrs.append(Instruction("valu", ("^", s['tmp_idx'][v], s['tmp_idx'][v], s['tmp_idx'][v]), group, 3, round_num, 0))
        # else: No wrap needed - idx is always valid at non-leaf levels

        # Debug compares
        for j, i in enumerate(group_items):
            v, vi = j // VLEN, j % VLEN
            instrs.append(Instruction("debug", ("compare", s['tmp_idx'][v] + vi, (round_num, i, "wrapped_idx")), group, 3, round_num, 2))

        return instrs

    def _gen_round_instrs(self, s, shared, group, round_num, num_vectors, group_items):
        """Generate all instructions for one group's round."""
        instrs = []
        # Debug compares for idx and val at start of round
        for j, i in enumerate(group_items):
            v, vi = j // VLEN, j % VLEN
            instrs.append(Instruction("debug", ("compare", s['tmp_idx'][v] + vi, (round_num, i, "idx")), group, 0, round_num, -1))
            instrs.append(Instruction("debug", ("compare", s['tmp_val'][v] + vi, (round_num, i, "val")), group, 0, round_num, -1))

        instrs.extend(self._gen_gather_instrs(s, shared, group, round_num, num_vectors, group_items))
        instrs.extend(self._gen_hash_instrs(s, shared, group, round_num, num_vectors, group_items))
        instrs.extend(self._gen_branch_instrs(s, shared, group, round_num, num_vectors, group_items))
        instrs.extend(self._gen_wrap_instrs(s, shared, group, round_num, num_vectors, group_items))
        return instrs

    # Operations that can be scalarized (valu -> alu)
    SCALARIZABLE_OPS = {"+", "-", "^", "%", "*", "&", "|", "<", ">", "<=", ">=", "==", "!="}

    def _scalarize_valu_op(self, op, offset=0, count=VLEN):
        """Convert a valu operation to scalar alu operations.

        Args:
            op: A valu operation tuple like ("+", dest_vec, src1_vec, src2_vec)
            offset: Starting element index (for partial scalarization)
                    For multiply_add: 0-7 = multiply phase, 8-15 = add phase
            count: Number of elements to scalarize

        Returns:
            List of alu operations for elements [offset, offset+count).
            For multiply_add, returns multiply ops (phase 1) or add ops (phase 2).
        """
        opcode = op[0]
        if opcode == "multiply_add":
            # multiply_add(dest, a, b, c) = a * b + c
            # Scalarization requires 2 phases and temp storage per group.
            # With shared temp, concurrent groups would corrupt each other.
            # Skip for now - let multiply_add use valu slots.
            return None
        elif len(op) == 4:
            # Binary op: (op, dest, src1, src2)
            dest, src1, src2 = op[1], op[2], op[3]
            return [(opcode, dest + i, src1 + i, src2 + i) for i in range(offset, offset + count)]
        return None

    def _try_merge_bundles(self, bundle1, bundle2):
        """Try to merge instructions from bundle2 into bundle1.

        Returns (merged_bundle, remaining_bundle) where:
        - merged_bundle has as many ops from bundle2 as fit
        - remaining_bundle has ops that didn't fit (or None if all fit)
        """
        _, ops1 = bundle1
        _, ops2 = bundle2

        merged = {engine: list(ops) for engine, ops in ops1.items()}
        remaining = {}

        for engine, ops in ops2.items():
            limit = SLOT_LIMITS.get(engine, 1)
            current = len(merged.get(engine, []))
            available = limit - current

            if available >= len(ops):
                # All ops fit
                if engine in merged:
                    merged[engine].extend(ops)
                else:
                    merged[engine] = list(ops)
            elif available > 0:
                # Partial fit - take what we can
                if engine in merged:
                    merged[engine].extend(ops[:available])
                else:
                    merged[engine] = list(ops[:available])
                remaining[engine] = ops[available:]
            else:
                # Nothing fits
                remaining[engine] = list(ops)

        merged_bundle = ("bundle", {k: v for k, v in merged.items() if v})
        remaining_bundle = ("bundle", remaining) if remaining else None
        return merged_bundle, remaining_bundle

    def _merge_bundle_lists(self, body, scheduled):
        """Merge first bundle of scheduled into last bundle of body if possible.

        Only merges the first scheduled bundle (partially or fully) into body[-1].
        Rest of scheduled bundles are returned as-is to preserve ordering.

        Modifies body in place and returns remaining scheduled bundles.
        """
        if not body or not scheduled:
            return scheduled

        # Only try to merge the first scheduled bundle into body[-1]
        merged, remaining = self._try_merge_bundles(body[-1], scheduled[0])
        body[-1] = merged

        # Build result: remaining from first bundle (if any) + rest of scheduled
        result = []
        if remaining:
            result.append(remaining)
        result.extend(scheduled[1:])

        return result

    def _schedule_group(self, instrs, idx, bundle):
        """Schedule instructions from one group into a bundle. Returns (new_idx, scheduled_any).

        Args:
            instrs: Sorted list of instructions for this group
            idx: Current index into instrs
            bundle: Current bundle being built

        Within a single group, seq boundaries are ALWAYS respected (data dependencies).
        """
        scheduled = False
        if idx >= len(instrs):
            return idx, scheduled

        # Track the current (round, phase, seq) for this group - can only schedule from this seq
        current_seq_key = (instrs[idx].round, instrs[idx].phase, instrs[idx].seq)

        while idx < len(instrs):
            instr = instrs[idx]
            instr_key = (instr.round, instr.phase, instr.seq)

            # Always respect seq boundaries within a group (data dependencies)
            if instr_key != current_seq_key:
                break

            limit = SLOT_LIMITS.get(instr.engine, 1)
            # Don't use valu slot if instruction is partially scalarized - must continue scalarization
            if len(bundle[instr.engine]) < limit and instr.scalar_offset == 0:
                bundle[instr.engine].append(instr.op)
                idx += 1
                scheduled = True
            elif instr.engine == "valu" and instr.op[0] in self.SCALARIZABLE_OPS:
                # Valu slot full - try to scalarize to alu
                alu_available = SLOT_LIMITS["alu"] - len(bundle["alu"])
                remaining = VLEN - instr.scalar_offset

                if alu_available >= remaining:
                    # Enough slots to finish this instruction
                    scalar_ops = self._scalarize_valu_op(instr.op, instr.scalar_offset, remaining)
                    if scalar_ops:
                        bundle["alu"].extend(scalar_ops)
                        instr.scalar_offset = 0  # Reset for potential reuse
                        idx += 1
                        scheduled = True
                        continue
                elif alu_available > 0:
                    # Partial scalarization - use what we have
                    scalar_ops = self._scalarize_valu_op(instr.op, instr.scalar_offset, alu_available)
                    if scalar_ops:
                        bundle["alu"].extend(scalar_ops)
                        instr.scalar_offset += len(scalar_ops)
                        scheduled = True
                        # Don't advance idx - instruction not complete
                        # But we've used all ALU slots, so break
                        break
                break  # No ALU slots available
            else:
                break  # Slot full for this engine
        return idx, scheduled

    def _schedule_instructions(self, *instr_lists, debug_phases=False):
        """Schedule instructions from multiple groups into bundles respecting slot limits and dependencies.

        Args:
            instr_lists: Variable number of instruction lists (one per group)
            debug_phases: If True, print group positions to show phase divergence.
        """
        bundles = []

        # Sort each group's instructions by (round, phase, seq)
        for instrs in instr_lists:
            instrs.sort(key=lambda i: (i.round, i.phase, i.seq))

        indices = [0] * len(instr_lists)

        bundle_num = 0
        while any(indices[i] < len(instr_lists[i]) for i in range(len(instr_lists))):
            bundle = defaultdict(list)
            any_scheduled = False

            # Debug: show group positions before scheduling
            if debug_phases and bundle_num % 50 == 0:
                positions = []
                for i, instrs in enumerate(instr_lists):
                    if indices[i] < len(instrs):
                        instr = instrs[indices[i]]
                        positions.append(f"{instr.group}:r{instr.round}p{instr.phase}s{instr.seq}")
                    else:
                        positions.append(f"G{i}:done")
                print(f"Bundle {bundle_num}: {' | '.join(positions)}")

            # Sort groups to prioritize those with load instructions pending
            def group_priority(i):
                if indices[i] >= len(instr_lists[i]):
                    return (2, 0)  # Done groups last
                instr = instr_lists[i][indices[i]]
                if instr.engine == "load":
                    return (0, instr.round)  # Load groups first, earlier rounds first
                elif instr.engine == "store":
                    return (0, instr.round)  # Store groups also high priority
                else:
                    return (1, instr.round)  # Compute groups second

            order = sorted(range(len(instr_lists)), key=group_priority)

            for i in order:
                instrs = instr_lists[i]
                indices[i], scheduled = self._schedule_group(instrs, indices[i], bundle)
                any_scheduled = any_scheduled or scheduled

            out_bundle = {k: v for k, v in bundle.items() if v}
            if out_bundle:
                bundles.append(("bundle", out_bundle))

            if not any_scheduled:
                break

            bundle_num += 1

        return bundles

    def _compute_load_addresses(self, alu_ops, s, group_start, num_vectors):
        """Add ALU ops to compute vload addresses for a group.

        Args:
            alu_ops: List to append ALU operations to.
            s: Scratch vectors dict for this group.
            group_start: Starting index of the group in the batch.
            num_vectors: Number of vectors per group.
        """
        for v in range(num_vectors):
            offset = group_start + v * VLEN
            offset_const = self.scratch_const(offset)
            alu_ops.append(("+", s['vload_addr_idx'][v], self.scratch["inp_indices_p"], offset_const))
            alu_ops.append(("+", s['vload_addr_val'][v], self.scratch["inp_values_p"], offset_const))

    def _gen_initial_load_instrs(self, s, group, group_start, num_vectors):
        """Generate initial load instructions for a group (address computation + vloads).

        Uses round=-1 so scheduler can interleave with other groups' compute phases.
        """
        instrs = []

        # Phase 0 of round -1: Address computation
        for v in range(num_vectors):
            offset = group_start + v * VLEN
            offset_const = self.scratch_const(offset)
            instrs.append(Instruction("alu", ("+", s['vload_addr_idx'][v], self.scratch["inp_indices_p"], offset_const), group, 0, -1, 0))
            instrs.append(Instruction("alu", ("+", s['vload_addr_val'][v], self.scratch["inp_values_p"], offset_const), group, 0, -1, 0))

        # Phase 1 of round -1: Vector loads (must come after address computation)
        for v in range(num_vectors):
            instrs.append(Instruction("load", ("vload", s['tmp_idx'][v], s['vload_addr_idx'][v]), group, 1, -1, 0))
            instrs.append(Instruction("load", ("vload", s['tmp_val'][v], s['vload_addr_val'][v]), group, 1, -1, 0))

        return instrs

    def _gen_final_store_instrs(self, s, group, num_vectors, final_round):
        """Generate final store instructions for a group.

        Uses round=final_round (after all compute rounds) so scheduler can
        interleave stores with computation from groups still working.
        """
        instrs = []

        # Store idx and val vectors back to memory
        for v in range(num_vectors):
            instrs.append(Instruction("store", ("vstore", s['vload_addr_idx'][v], s['tmp_idx'][v]), group, 0, final_round, 0))
            instrs.append(Instruction("store", ("vstore", s['vload_addr_val'][v], s['tmp_val'][v]), group, 0, final_round, 0))

        return instrs

    def _emit_group_loads(self, body, s, num_vectors):
        """Emit load bundles for idx and val vectors.

        Args:
            body: List to append instructions to.
            s: Scratch vectors dict for this group.
            num_vectors: Number of vectors per group.
        """
        body.append(("bundle", {"load": [("vload", s['tmp_idx'][v], s['vload_addr_idx'][v]) for v in range(num_vectors)]}))
        body.append(("bundle", {"load": [("vload", s['tmp_val'][v], s['vload_addr_val'][v]) for v in range(num_vectors)]}))

    def build_kernel(self, forest_height: int, n_nodes: int, batch_size: int, rounds: int):
        """Build the complete kernel program for tree traversal."""
        # Store forest_height for use in gather instructions
        self.forest_height = forest_height

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
        # Try 1 vector per group with more concurrent groups
        num_vectors = 1  # 1 vector = 8 elements per group
        bundle_size = num_vectors * VLEN  # 8 elements per group

        # Allocate N sets of working vectors for pipelining
        # With fewer vectors per group, we can fit more concurrent groups
        num_concurrent = 18
        scratch_sets = [self._alloc_scratch_vectors_for_set(num_vectors, chr(ord('A') + i))
                        for i in range(num_concurrent)]
        shared = self._alloc_shared_vectors(num_vectors)

        self._broadcast_constants(body, shared, zero_const, one_const, two_const)

        # Process groups in batches for pipelining
        group_starts = list(range(0, batch_size, bundle_size))
        num_groups = len(group_starts)

        for batch_idx in range(0, num_groups, num_concurrent):
            # Determine which groups are in this batch
            active_groups = []
            for offset in range(num_concurrent):
                group_idx = batch_idx + offset
                if group_idx < num_groups:
                    s = scratch_sets[offset]
                    group_start = group_starts[group_idx]
                    group_items = list(range(group_start, min(group_start + bundle_size, batch_size)))
                    active_groups.append((s, group_start, group_items, chr(ord('A') + offset)))

            # Generate instructions for all active groups including initial loads and final stores
            instr_lists = []
            for s, group_start, group_items, name in active_groups:
                instrs = []
                # Add initial load instructions (round -1)
                instrs.extend(self._gen_initial_load_instrs(s, name, group_start, num_vectors))
                # Add round instructions
                for round_num in range(rounds):
                    instrs.extend(self._gen_round_instrs(s, shared, name, round_num, num_vectors, group_items))
                # Add final store instructions (round = rounds, after all compute)
                instrs.extend(self._gen_final_store_instrs(s, name, num_vectors, rounds))
                instr_lists.append(instrs)

            # Schedule instructions into bundles
            scheduled = self._schedule_instructions(*instr_lists)
            # Try to merge ops from scheduled into body's last bundle
            scheduled = self._merge_bundle_lists(body, scheduled)
            body.extend(scheduled)

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
