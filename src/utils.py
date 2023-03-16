def round_prediction(prediction: float) -> int:
    '''
    Round and clamp the prediction.
    '''
    rounded = round(prediction)
    if rounded < 1:
        return 1
    elif rounded > 5:
        return 5
    return rounded
