import unittest
from load_and_Represent_graph import main


class TestIntegrationPipeline(unittest.TestCase):
    def test_full_pipeline(self):
        # Run the main script
        main()

        # Check for output files
        import glob
        result_files = glob.glob("results_*.tsv")
        graph_files = glob.glob("graph_*.png")
        eval_files = glob.glob("evaluation_*.txt")
        performance_files = glob.glob("performance_*.txt")

        self.assertTrue(len(result_files) > 0, "Results file not created.")
        self.assertTrue(len(graph_files) > 0, "Graph file not created.")
        self.assertTrue(len(eval_files) > 0, "Evaluation file not created.")
        self.assertTrue(len(performance_files) > 0, "Performance file not created.")
