"""CPU release checks; no EasyFHE/CUDA import or synthetic speed claims."""
import importlib
from pathlib import Path
import numpy as np
import pytest

from baselines.sc.run import build_parser, make_config
from baselines.sc.rotation import OrthogonalRotation, convert_weights


@pytest.mark.parametrize('model,count', [('1b',16),('3b',28),('8b',32)])
@pytest.mark.parametrize('world', [1,2,4,8])
@pytest.mark.parametrize('phase', ['prefill','decode','linked'])
def test_portable_plan(tmp_path, model, count, world, phase):
    args = build_parser().parse_args(['--model', model, '--world-size', str(world),
        '--phase',phase,'--assets-root',str(tmp_path),'--output',str(tmp_path/'out')])
    config=make_config(args)
    assert config['layers']==count
    assert Path(config['chain'])==tmp_path/'sc'/model/'chain'
    provider=Path(config['sylph_root'])
    assert (provider/'sylph_easy/layer/decoder.py').is_file()
    assert (provider/'sylph_easy/easyfhe_ext/kernels/keyswitch.cu').is_file()


def test_sc_basis_conversion_preserves_full_plaintext_decoder():
    plain=importlib.import_module('baselines.sc.providers.llama8b.sylph_easy.layer.plain')
    rng=np.random.default_rng(5)
    h,heads,kv,dim,ffn=8,2,1,4,16
    w={name:rng.normal(scale=.15,size=shape) for name,shape in {
        'q':(h,h),'k':(kv*dim,h),'v':(kv*dim,h),'o':(h,h),
        'gate':(ffn,h),'up':(ffn,h),'down':(h,ffn)}.items()}
    w.update(attention_norm=rng.uniform(.5,1.5,h),ffn_norm=rng.uniform(.5,1.5,h),
             public_keys=np.zeros((0,kv,dim)),public_values=np.zeros((0,kv,dim)))
    r=OrthogonalRotation.generate(h,20260908)
    v=OrthogonalRotation.generate(dim,20270908)
    converted=convert_weights(w,r,v,hidden=h,heads=heads,kv_heads=kv,head_dim=dim)
    converted.update(public_keys=w['public_keys'],public_values=w['public_values'])
    x=rng.normal(size=(5,h))
    want,_=plain.layer(x,w,heads=heads,kv_heads=kv,head_dim=dim)
    actual,_=plain.layer(r.apply(x),converted,heads=heads,kv_heads=kv,head_dim=dim)
    np.testing.assert_allclose(r.apply(actual,inverse=True),want,rtol=1e-12,atol=1e-12)
