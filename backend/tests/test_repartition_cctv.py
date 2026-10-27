from backend.scripts.repartition_cctv import extract_room_prefix, filter_phone_labels


def test_extract_room_prefix():
    assert extract_room_prefix("Class_1_Video_1_mp4-0_jpg.rf.bb97.jpg") == "Class_1"
    assert extract_room_prefix("Class_4_Video_2_mp4-15_jpg.rf.1234.jpg") == "Class_4"


def test_filter_phone_labels_keeps_only_class_zero():
    raw_lines = [
        "0 0.455 0.552 0.021 0.026\n",
        "1 0.410 0.433 0.066 0.147\n",
        "0 0.372 0.454 0.018 0.028\n",
    ]
    filtered = filter_phone_labels(raw_lines)
    assert len(filtered) == 2
    assert all(line.startswith("0 ") for line in filtered)
