from backend.scripts.evaluate_pipeline import iou_xyxy, accumulate_pr


def test_iou_xyxy_and_accumulate_pr():
    gt = [(100.0, 100.0, 200.0, 200.0)]
    preds = [(110.0, 110.0, 200.0, 200.0), (500.0, 500.0, 550.0, 550.0)]
    assert iou_xyxy(gt[0], preds[0]) > 0.6

    counter = {"tp": 0, "fp": 0, "fn": 0}
    accumulate_pr(gt, preds, counter, iou_thres=0.35)
    assert counter == {"tp": 1, "fp": 1, "fn": 0}
