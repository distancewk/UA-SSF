import unittest

from runtime_config import resolve_torch_device


class RuntimeConfigTests(unittest.TestCase):
    def test_cpu_cuda_argument_forces_cpu_even_when_cuda_is_available(self):
        self.assertEqual(resolve_torch_device("cpu", cuda_available=True), "cpu")

    def test_numeric_cuda_argument_uses_gpu_only_when_available(self):
        self.assertEqual(resolve_torch_device("0", cuda_available=True), "cuda:0")
        self.assertEqual(resolve_torch_device("0", cuda_available=False), "cpu")


if __name__ == "__main__":
    unittest.main()
