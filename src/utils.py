def round_prediction(prediction: float) -> int:
    '''
    Round and clamp the prediction.
    '''
    rounded = round(prediction)
    if rounded == 0:
        return 1
    elif rounded == 6:
        return 5
    return rounded
