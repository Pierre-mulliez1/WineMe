def test_executiontime():
    # test the execution time of training
    t0 = time.clock()
    code_to_test = train_and_persist()
    elapsed_time = time.clock() - t0

    assert float(elapsed_time) < 10.0
