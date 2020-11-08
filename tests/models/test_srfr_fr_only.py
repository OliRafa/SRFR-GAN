from models.srfr_fr_only import SrfrFrOnly


def test_1(ground_truth_images):
    fr = SrfrFrOnly()
    output = fr(ground_truth_images)
    print(output)
