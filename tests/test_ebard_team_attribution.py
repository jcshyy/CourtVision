import unittest

from scripts.evaluate_ebard_fashionclip import classification_metrics, feature_tensor


class EbardTeamAttributionTests(unittest.TestCase):
    def test_feature_tensor_accepts_transformers_five_output(self):
        class Output:
            pooler_output = "pooled"

        self.assertEqual(feature_tensor(Output()), "pooled")

    def test_classification_metrics_include_false_referee_predictions(self):
        metrics = classification_metrics(
            ["blue", "blue", "white"],
            ["blue", "referee", "white"],
        )

        self.assertEqual(metrics["accuracy"], 0.666667)
        self.assertEqual(metrics["per_class"]["blue"]["recall"], 0.5)
        self.assertEqual(metrics["per_class"]["referee"]["support"], 0)

    def test_classification_metrics_build_confusion_matrix(self):
        metrics = classification_metrics(
            ["blue", "white"],
            ["white", "white"],
        )

        self.assertEqual(metrics["confusion_matrix"]["blue"]["white"], 1)
        self.assertEqual(metrics["per_class"]["white"]["precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
