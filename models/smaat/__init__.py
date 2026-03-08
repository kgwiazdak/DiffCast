from .smaat import SmaAtNowcastBackbone


def get_model(in_shape, T_in, T_out, **kwargs):
    return SmaAtNowcastBackbone(in_shape=in_shape, T_in=T_in, T_out=T_out, **kwargs)
