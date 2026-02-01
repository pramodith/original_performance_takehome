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

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """Build the complete kernel program for tree traversal.

        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.

        Args:
            forest_height: Height of the binary tree.
            n_nodes: Total number of nodes in the tree.
            batch_size: Number of parallel traversals in each batch.
            rounds: Number of traversal rounds to perform.
        """
        # Scratch space addresses for init vars
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        tmp_init = self.alloc_scratch("tmp_init")
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp_init, i))
            self.add("load", ("load", self.scratch[v], tmp_init))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Allocate scratch register arrays for bundling
        # Size based on the bottleneck operation (load/store with limit 2)
        bundle_size = SLOT_LIMITS["load"]  # Use load limit as bundle size

        # Allocate arrays of scratch registers for bundled operations
        tmp_idx = [self.alloc_scratch(f"tmp_idx_{j}") for j in range(bundle_size)]
        tmp_val = [self.alloc_scratch(f"tmp_val_{j}") for j in range(bundle_size)]
        tmp_node_val = [self.alloc_scratch(f"tmp_node_val_{j}") for j in range(bundle_size)]
        tmp_addr = [self.alloc_scratch(f"tmp_addr_{j}") for j in range(bundle_size)]
        tmp1 = [self.alloc_scratch(f"tmp1_{j}") for j in range(bundle_size)]
        tmp2 = [self.alloc_scratch(f"tmp2_{j}") for j in range(bundle_size)]
        tmp3 = [self.alloc_scratch(f"tmp3_{j}") for j in range(bundle_size)]

        for round in range(rounds):
            # Process batch in groups of bundle_size
            for group_start in range(0, batch_size, bundle_size):
                group_end = min(group_start + bundle_size, batch_size)
                group_items = list(range(group_start, group_end))

                # idx = mem[inp_indices_p + i] - bundled ALU + load
                alu_ops = []
                for j, i in enumerate(group_items):
                    i_const = self.scratch_const(i)
                    alu_ops.append(("+", tmp_addr[j], self.scratch["inp_indices_p"], i_const))
                body.append(("bundle", {"alu": alu_ops}))

                load_ops = []
                for j, i in enumerate(group_items):
                    load_ops.append(("load", tmp_idx[j], tmp_addr[j]))
                body.append(("bundle", {"load": load_ops}))

                for j, i in enumerate(group_items):
                    body.append(("debug", ("compare", tmp_idx[j], (round, i, "idx"))))

                # val = mem[inp_values_p + i] - bundled ALU + load
                alu_ops = []
                for j, i in enumerate(group_items):
                    i_const = self.scratch_const(i)
                    alu_ops.append(("+", tmp_addr[j], self.scratch["inp_values_p"], i_const))
                body.append(("bundle", {"alu": alu_ops}))

                load_ops = []
                for j, i in enumerate(group_items):
                    load_ops.append(("load", tmp_val[j], tmp_addr[j]))
                body.append(("bundle", {"load": load_ops}))

                for j, i in enumerate(group_items):
                    body.append(("debug", ("compare", tmp_val[j], (round, i, "val"))))

                # node_val = mem[forest_values_p + idx] - bundled ALU + load
                alu_ops = []
                for j, i in enumerate(group_items):
                    alu_ops.append(("+", tmp_addr[j], self.scratch["forest_values_p"], tmp_idx[j]))
                body.append(("bundle", {"alu": alu_ops}))

                load_ops = []
                for j, i in enumerate(group_items):
                    load_ops.append(("load", tmp_node_val[j], tmp_addr[j]))
                body.append(("bundle", {"load": load_ops}))

                for j, i in enumerate(group_items):
                    body.append(("debug", ("compare", tmp_node_val[j], (round, i, "node_val"))))

                # val = myhash(val ^ node_val) - bundled ALU
                alu_ops = []
                for j, i in enumerate(group_items):
                    alu_ops.append(("^", tmp_val[j], tmp_val[j], tmp_node_val[j]))
                body.append(("bundle", {"alu": alu_ops}))

                # Hash stages - bundled
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    alu_ops = []
                    for j, i in enumerate(group_items):
                        alu_ops.append((op1, tmp1[j], tmp_val[j], self.scratch_const(val1)))
                    body.append(("bundle", {"alu": alu_ops}))

                    alu_ops = []
                    for j, i in enumerate(group_items):
                        alu_ops.append((op3, tmp2[j], tmp_val[j], self.scratch_const(val3)))
                    body.append(("bundle", {"alu": alu_ops}))

                    alu_ops = []
                    for j, i in enumerate(group_items):
                        alu_ops.append((op2, tmp_val[j], tmp1[j], tmp2[j]))
                    body.append(("bundle", {"alu": alu_ops}))

                    for j, i in enumerate(group_items):
                        body.append(("debug", ("compare", tmp_val[j], (round, i, "hash_stage", hi))))

                for j, i in enumerate(group_items):
                    body.append(("debug", ("compare", tmp_val[j], (round, i, "hashed_val"))))

                # idx = 2*idx + (1 if val % 2 == 0 else 2) - bundled ops
                alu_ops = []
                for j, i in enumerate(group_items):
                    alu_ops.append(("%", tmp1[j], tmp_val[j], two_const))
                body.append(("bundle", {"alu": alu_ops}))

                alu_ops = []
                for j, i in enumerate(group_items):
                    alu_ops.append(("==", tmp1[j], tmp1[j], zero_const))
                body.append(("bundle", {"alu": alu_ops}))

                for j, i in enumerate(group_items):
                    body.append(("flow", ("select", tmp3[j], tmp1[j], one_const, two_const)))

                alu_ops = []
                for j, i in enumerate(group_items):
                    alu_ops.append(("*", tmp_idx[j], tmp_idx[j], two_const))
                body.append(("bundle", {"alu": alu_ops}))

                alu_ops = []
                for j, i in enumerate(group_items):
                    alu_ops.append(("+", tmp_idx[j], tmp_idx[j], tmp3[j]))
                body.append(("bundle", {"alu": alu_ops}))

                for j, i in enumerate(group_items):
                    body.append(("debug", ("compare", tmp_idx[j], (round, i, "next_idx"))))

                # idx = 0 if idx >= n_nodes else idx - bundled ops
                alu_ops = []
                for j, i in enumerate(group_items):
                    alu_ops.append(("<", tmp1[j], tmp_idx[j], self.scratch["n_nodes"]))
                body.append(("bundle", {"alu": alu_ops}))

                for j, i in enumerate(group_items):
                    body.append(("flow", ("select", tmp_idx[j], tmp1[j], tmp_idx[j], zero_const)))

                for j, i in enumerate(group_items):
                    body.append(("debug", ("compare", tmp_idx[j], (round, i, "wrapped_idx"))))

                # mem[inp_indices_p + i] = idx - bundled ops
                alu_ops = []
                for j, i in enumerate(group_items):
                    i_const = self.scratch_const(i)
                    alu_ops.append(("+", tmp_addr[j], self.scratch["inp_indices_p"], i_const))
                body.append(("bundle", {"alu": alu_ops}))

                store_ops = []
                for j, i in enumerate(group_items):
                    store_ops.append(("store", tmp_addr[j], tmp_idx[j]))
                body.append(("bundle", {"store": store_ops}))

                # mem[inp_values_p + i] = val - bundled ops
                alu_ops = []
                for j, i in enumerate(group_items):
                    i_const = self.scratch_const(i)
                    alu_ops.append(("+", tmp_addr[j], self.scratch["inp_values_p"], i_const))
                body.append(("bundle", {"alu": alu_ops}))

                store_ops = []
                for j, i in enumerate(group_items):
                    store_ops.append(("store", tmp_addr[j], tmp_val[j]))
                body.append(("bundle", {"store": store_ops}))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
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
