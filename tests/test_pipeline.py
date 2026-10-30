from shredded_reconstruction.document import generate_document
from shredded_reconstruction.evaluation import order_metrics, run_case
from shredded_reconstruction.learned import train_compatibility_model
from shredded_reconstruction.shred import shred_document


def test_metrics_use_labels_only_after_prediction():
    result = order_metrics([1, 2, 0], [2, 0, 1])
    assert result["predicted_original_order"] == [0, 1, 2]
    assert result["exact_match"] is True
    assert result["adjacency_accuracy"] == 1.0


def test_baseline_end_to_end_returns_permutation():
    result = run_case(seed=501, variant="table", strip_count=8, shuffle_seed=99)
    assert sorted(result.baseline_order) == list(range(8))
    assert result.baseline_reconstruction.size == result.original.size
    assert 0.0 <= result.baseline_metrics["adjacency_accuracy"] <= 1.0


def test_learned_model_trains_on_different_documents():
    training_specs = [(701, "report", 8), (702, "table", 9), (703, "mixed", 10)]
    model = train_compatibility_model(training_specs)
    assert all("999" not in item for item in model.training_document_ids)
    image = generate_document(seed=999, variant="mixed")
    shredded = shred_document(image, strip_count=8, shuffle_seed=5)
    order, _ = model.reconstruct(shredded.strips)
    assert sorted(order) == list(range(8))
