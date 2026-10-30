from PIL import Image

from shredded_reconstruction.shred import compose_strips, cut_vertical_strips, shred_document


def test_cut_and_correct_compose_are_pixel_exact():
    image = Image.new("RGB", (101, 40))
    for x in range(image.width):
        for y in range(image.height):
            image.putpixel((x, y), (x, y, (x + y) % 256))
    strips = cut_vertical_strips(image, 9)
    restored = compose_strips(strips)
    assert restored.size == image.size
    assert restored.tobytes() == image.tobytes()


def test_shuffle_labels_are_evaluation_metadata_only():
    image = Image.new("RGB", (96, 32), "white")
    shredded = shred_document(image, 8, shuffle_seed=7)
    assert sorted(shredded.original_indices) == list(range(8))
    assert list(shredded.original_indices) != list(range(8))
    assert len(shredded.strips) == 8
