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

    def _alloc_scratch_vectors(self, num_vectors):
        """Allocate all scratch vectors needed for computation.

        Returns a dict with all allocated scratch vector addresses.
        """
        s = {}
        # Working vectors for idx, val, node_val
        s['tmp_idx'] = [self.alloc_scratch(f"tmp_idx_{j}", VLEN) for j in range(num_vectors)]
        s['tmp_val'] = [self.alloc_scratch(f"tmp_val_{j}", VLEN) for j in range(num_vectors)]
        s['tmp_node_val'] = [self.alloc_scratch(f"tmp_node_val_{j}", VLEN) for j in range(num_vectors)]
        s['tmp_addr'] = [self.alloc_scratch(f"tmp_addr_{j}", VLEN) for j in range(num_vectors)]

        # Temporary vectors for computation
        s['tmp1'] = [self.alloc_scratch(f"tmp1_{j}", VLEN) for j in range(num_vectors)]
        s['tmp2'] = [self.alloc_scratch(f"tmp2_{j}", VLEN) for j in range(num_vectors)]
        s['tmp3'] = [self.alloc_scratch(f"tmp3_{j}", VLEN) for j in range(num_vectors)]
        s['idx_plus_1'] = [self.alloc_scratch(f"idx_plus_1_{j}", VLEN) for j in range(num_vectors)]
        s['idx_plus_2'] = [self.alloc_scratch(f"idx_plus_2_{j}", VLEN) for j in range(num_vectors)]

        # Scalar addresses for vload/vstore
        s['vload_addr'] = [self.alloc_scratch(f"vload_addr_{j}") for j in range(num_vectors)]
        s['vload_addr_val'] = [self.alloc_scratch(f"vload_addr_val_{v}") for v in range(num_vectors)]

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

    def _broadcast_constants(self, body, s, zero_const, one_const, two_const):
        """Broadcast all constant vectors once before the main loop."""
        # Basic constants
        body.append(("bundle", {"valu": [
            ("vbroadcast", s['two_vec'], two_const),
            ("vbroadcast", s['zero_vec'], zero_const),
            ("vbroadcast", s['one_vec'], one_const),
            ("vbroadcast", s['n_nodes_vec'], self.scratch["n_nodes"]),
            ("vbroadcast", s['forest_values_p_vec'], self.scratch["forest_values_p"]),
        ]}))

        # Hash stage constants (split into two bundles, valu limit is 6)
        valu_ops = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES[:3]):
            valu_ops.append(("vbroadcast", s['val1_vecs'][hi], self.scratch_const(val1)))
            valu_ops.append(("vbroadcast", s['val3_vecs'][hi], self.scratch_const(val3)))
        body.append(("bundle", {"valu": valu_ops}))

        valu_ops = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES[3:], start=3):
            valu_ops.append(("vbroadcast", s['val1_vecs'][hi], self.scratch_const(val1)))
            valu_ops.append(("vbroadcast", s['val3_vecs'][hi], self.scratch_const(val3)))
        body.append(("bundle", {"valu": valu_ops}))

    def _build_gather_node_values(self, body, s, group_items, num_vectors):
        """Build gather operation to load node values from tree (non-contiguous addresses).

        Overlaps XOR of vector 0 with loading of vector 1 to hide XOR latency.
        XOR of vector 1 is done after all loads complete.
        """
        # Compute node_val addresses: forest_values_p + idx
        valu_ops = []
        for v in range(num_vectors):
            valu_ops.append(("+", s['tmp_addr'][v], s['tmp_idx'][v], s['forest_values_p_vec']))
        body.append(("bundle", {"valu": valu_ops}))

        # Individual loads in groups of 2 (load slot limit)
        # After vector 0 is complete (4 cycles), XOR it while continuing to load vector 1
        loads_per_vector = VLEN // SLOT_LIMITS["load"]  # 4 cycles to load one vector

        for load_cycle, load_start in enumerate(range(0, len(group_items), SLOT_LIMITS["load"])):
            load_end = min(load_start + SLOT_LIMITS["load"], len(group_items))
            load_ops = []
            for li in range(load_start, load_end):
                v, vi = li // VLEN, li % VLEN
                load_ops.append(("load", s['tmp_node_val'][v] + vi, s['tmp_addr'][v] + vi))

            # On cycle 4 (first cycle of loading vector 1), XOR the completed vector 0
            if load_cycle > 0 and load_cycle % loads_per_vector == 0:
                valu_ops = [("^", s['tmp_val'][v-1], s['tmp_val'][v-1], s['tmp_node_val'][v-1])]
                body.append(("bundle", {"load": load_ops, "valu": valu_ops}))
            else:
                body.append(("bundle", {"load": load_ops}))

        valu_ops = [("^", s['tmp_val'][num_vectors-1], s['tmp_val'][num_vectors-1], s['tmp_node_val'][num_vectors-1])]
        body.append(("bundle", {"valu": valu_ops}))

    def _build_hash_stages(self, body, s, num_vectors, group_items, round):
        """Build the 6-stage hash computation: val = myhash(val ^ node_val).

        All XORs are done in the gather phase before this function is called.
        """
        # Apply 6 hash stages (constants already broadcast)
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            # Fused: tmp1 = val op1 val1, tmp2 = val op3 val3 (independent)
            valu_ops = []
            for v in range(num_vectors):
                valu_ops.append((op1, s['tmp1'][v], s['tmp_val'][v], s['val1_vecs'][hi]))
                valu_ops.append((op3, s['tmp2'][v], s['tmp_val'][v], s['val3_vecs'][hi]))
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

    def _build_branch_computation(self, body, s, num_vectors):
        """Build: idx = 2*idx + (1 if val%2==0 else 2)."""
        # Fused: tmp1 = val % 2, idx_plus_1 = 2*idx+1, idx_plus_2 = 2*idx+2
        valu_ops = []
        for v in range(num_vectors):
            valu_ops.append(("%", s['tmp1'][v], s['tmp_val'][v], s['two_vec']))
            valu_ops.append(("multiply_add", s['idx_plus_1'][v], s['tmp_idx'][v], s['two_vec'], s['one_vec']))
            valu_ops.append(("multiply_add", s['idx_plus_2'][v], s['tmp_idx'][v], s['two_vec'], s['two_vec']))
        body.append(("bundle", {"valu": valu_ops}))

        # Select: if even (tmp1=0) pick idx_plus_1, if odd (tmp1=1) pick idx_plus_2
        for v in range(num_vectors):
            body.append(("flow", ("vselect", s['tmp_idx'][v], s['tmp1'][v], s['idx_plus_2'][v], s['idx_plus_1'][v])))

    def _build_wrap_check(self, body, s, num_vectors):
        """Build: idx = 0 if idx >= n_nodes else idx."""
        # tmp1 = idx < n_nodes
        valu_ops = []
        for v in range(num_vectors):
            valu_ops.append(("<", s['tmp1'][v], s['tmp_idx'][v], s['n_nodes_vec']))
        body.append(("bundle", {"valu": valu_ops}))

        # idx = select(tmp1, idx, 0)
        for v in range(num_vectors):
            body.append(("flow", ("vselect", s['tmp_idx'][v], s['tmp1'][v], s['tmp_idx'][v], s['zero_vec'])))

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
        s = self._alloc_scratch_vectors(num_vectors)
        self._broadcast_constants(body, s, zero_const, one_const, two_const)

        # Main loop: process each group across all rounds
        for group_start in range(0, batch_size, bundle_size):
            group_end = min(group_start + bundle_size, batch_size)
            group_items = list(range(group_start, group_end))

            # Compute load/store addresses for this group
            alu_ops, idx_addrs, val_addrs = [], [], []
            for v in range(num_vectors):
                offset = group_start + v * VLEN
                self.add_offset_ops(alu_ops, idx_addrs, self.scratch["inp_indices_p"], offset, s['vload_addr'][v])
                self.add_offset_ops(alu_ops, val_addrs, self.scratch["inp_values_p"], offset, s['vload_addr_val'][v])
            if alu_ops:
                body.append(("bundle", {"alu": alu_ops}))

            # Load idx and val once at start of group
            body.append(("bundle", {"load": [("vload", s['tmp_idx'][v], idx_addrs[v]) for v in range(num_vectors)]}))
            body.append(("bundle", {"load": [("vload", s['tmp_val'][v], val_addrs[v]) for v in range(num_vectors)]}))

            # Process all rounds for this group
            for round in range(rounds):
                self._build_debug_compares(body, s, group_items, round, "idx")
                self._build_debug_compares(body, s, group_items, round, "val")

                # Gather node values from tree
                self._build_gather_node_values(body, s, group_items, num_vectors)
                self._build_debug_compares(body, s, group_items, round, "node_val")

                # Hash computation
                self._build_hash_stages(body, s, num_vectors, group_items, round)
                self._build_debug_compares(body, s, group_items, round, "hashed_val")

                # Branch: idx = 2*idx + (1 or 2)
                self._build_branch_computation(body, s, num_vectors)
                self._build_debug_compares(body, s, group_items, round, "next_idx")

                # Wrap check: idx = 0 if idx >= n_nodes
                self._build_wrap_check(body, s, num_vectors)
                self._build_debug_compares(body, s, group_items, round, "wrapped_idx")

            # Store idx and val once at end of group
            body.append(("bundle", {"store": [("vstore", idx_addrs[v], s['tmp_idx'][v]) for v in range(num_vectors)]}))
            body.append(("bundle", {"store": [("vstore", val_addrs[v], s['tmp_val'][v]) for v in range(num_vectors)]}))

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
