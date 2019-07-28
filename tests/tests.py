import unittest
import tindetheus  # noqa F401


class TestEverything(unittest.TestCase):

    def test_imports(self):
        from tindetheus import export_embeddings  # noqa F401
        from tindetheus import tindetheus_align  # noqa F401
        from tindetheus.tinder_client import client  # noqa F401
        import tindetheus.facenet_clone.facenet as facenet  # noqa F401
        import tindetheus.image_processing as imgproc  # noqa F401
        import tindetheus.machine_learning as ml  # noqa F401
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
