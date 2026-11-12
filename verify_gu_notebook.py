import ast, json, pathlib, torch, uuid, shutil, types, sys, zipfile, warnings
from unittest.mock import PropertyMock, patch
p=pathlib.Path(__file__).with_name('training_Gu_1B.ipynb')
n=json.loads(p.read_text(encoding='utf-8'))
e={}
for i,c in enumerate(n['cells']):
    if c['cell_type']=='code':
        s=''.join(c['source'])
        ast.parse('\n'.join(line for line in s.splitlines() if not line.startswith('%')))
        if s.startswith('%pip'):
            s=s[s.index('import os,'):s.index('SAVE_TO_DRIVE =')]
            exec(s,e)
        elif s.startswith(('@dataclass','class RMSNorm','def source_metadata','def versions','def train_gu')):
            exec(s,e)
assert all(c['cell_type']=='markdown' for c in n['cells'][:5])
print('PASS all code cells parse and the first five cells are Markdown')
torch.set_num_threads(2)
C=e['ModelConfig']; R=e['RunConfig']; M=e['GuModel']
c=C(vocab_size=64,d_model=16,n_layers=1,n_heads=2,n_kv_heads=1,d_ff=32,max_seq_len=16,loss_chunk=4)
x=torch.randint(0,64,(1,16)); y=torch.randint(0,64,(1,16))
m=M(c)
full=torch.nn.functional.cross_entropy(m(x).reshape(-1,64),y.reshape(-1),reduction='sum')
chunk=m.loss_sum(x,y)
torch.testing.assert_close(full,chunk)
full.backward(); a=[v.grad.clone() for v in m.parameters()]; m.zero_grad(); m.loss_sum(x,y).backward()
for g,v in zip(a,m.parameters()): torch.testing.assert_close(g,v.grad,rtol=1e-4,atol=1e-5)
with torch.no_grad():
    before=m(x)[:,:8]; z=x.clone(); z[:,8:]=(z[:,8:]+1)%64
    torch.testing.assert_close(before,m(z)[:,:8],rtol=1e-5,atol=1e-6)
with torch.autocast('cpu', dtype=torch.bfloat16):
    bf16_loss=m.loss_sum(x,y)
assert torch.isfinite(bf16_loss)
bf16_loss.backward()
print('PASS causal attention, BF16 forward/backward and recomputed chunk-loss gradients')
# Route CPU tensors through the CUDA policy to check that math remains enabled.
original_attention = e['F'].scaled_dot_product_attention
def require_math_fallback(*args, **kwargs):
    assert torch.backends.cuda.math_sdp_enabled()
    return original_attention(*args, **kwargs)
with patch.object(torch.Tensor, 'is_cuda', new_callable=PropertyMock, return_value=True), \
     patch.object(e['F'], 'scaled_dot_product_attention', side_effect=require_math_fallback):
    with torch.inference_mode():
        fallback_logits = m.eval()(x)
assert torch.isfinite(fallback_logits).all()
print('PASS attention policy permits math fallback when CUDA FlashAttention is unavailable (simulated)')
class Data:
    def __init__(self,metadata): self.metadata=metadata; self.pos=0
    def batch(self,count,device):
        ids=torch.arange(self.pos,self.pos+count+1,device=device).remainder(64)[None,:]
        self.pos+=count
        return ids[:,:-1],ids[:,1:]
    def state_dict(self): return {'pos':self.pos}
    def load_state_dict(self,s): self.pos=s['pos']
e['source_metadata']=lambda:{'tokenizer_id':'offline-test','tokenizer_revision':'test'}
root=pathlib.Path.cwd()/('.gu-test-'+uuid.uuid4().hex)
root.mkdir()
try:
    r=R(target_tokens=64,update_tokens=8,checkpoint_every=16,resume_every=32,warmup_tokens=8,stage_ends=(24,48),stage_lengths=(4,8,16))
    baseline,t=e['train_gu'](c,r,root/'baseline',device='cpu',data_factory=Data)
    partial,t=e['train_gu'](c,r,root/'resumed',session_stop=32,device='cpu',data_factory=Data)
    assert t==32
    source=e['latest_full'](root/'resumed')
    resumed,t=e['train_gu'](c,r,root/'resumed',resume_from=source,device='cpu',data_factory=Data)
    assert t==64
    for a,b in zip(baseline.parameters(),resumed.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
    checkpoints=e['checkpoint_files'](root/'baseline')
    assert [(i['tokens'],i['resumable']) for i,p in checkpoints]==[(16,False),(32,True),(48,False),(64,True)]
    lightweight=torch.load(checkpoints[0][1],weights_only=True)
    assert all(v.dtype==torch.bfloat16 for v in lightweight['weights'].values())
    continued,t=e['train_gu'](c,R(**{**r.__dict__,'target_tokens':80}),root/'resumed',resume_from=e['latest_full'](root/'resumed'),device='cpu',data_factory=Data)
    assert t==80
    print('PASS exact milestones, BF16 exports, bitwise resumed training, budget extension')
    changed_config = C(**{**c.__dict__, 'loss_chunk': 8})
    shorter_run = R(**{**r.__dict__, 'target_tokens': 56, 'stage_ends': (24, 40)})
    retuned, t = e['train_gu'](changed_config,shorter_run,root/'retuned',
                              resume_from=source,device='cpu',data_factory=Data)
    assert t == 56
    retuned_save = torch.load(e['latest_full'](root/'retuned'),weights_only=False)
    assert retuned_save['config']['loss_chunk'] == 8
    assert retuned_save['run']['stage_ends'] == (24, 40)
    assert retuned_save['data']['pos'] == 56
    for bad_config, bad_run, message in [
        (C(**{**c.__dict__, 'd_ff': 64}), shorter_run, 'Model settings changed'),
        (changed_config, R(**{**shorter_run.__dict__, 'stage_ends': (20, 40)}), 'before saved progress'),
        (changed_config, R(**{**shorter_run.__dict__, 'stage_ends': (24, 30)}), 'before saved progress'),
    ]:
        try:
            e['train_gu'](bad_config,bad_run,root/'invalid',resume_from=source,
                          device='cpu',data_factory=Data)
        except ValueError as error:
            assert message in str(error), error
        else:
            raise AssertionError('Incompatible checkpoint settings were accepted')
    print('PASS shortened target, changed loss chunks and future stages resume; incompatible changes rejected')
    uneven = R(target_tokens=34,update_tokens=11,checkpoint_every=7,resume_every=14,
               warmup_tokens=8,stage_ends=(17,26),stage_lengths=(4,8,16))
    _, t = e['train_gu'](c,uneven,root/'uneven',device='cpu',data_factory=Data)
    assert t == 34
    assert [(i['tokens'],i['resumable']) for i,_ in e['checkpoint_files'](root/'uneven')] == [
        (7,False),(14,True),(21,False),(28,True),(34,True)]
    print('PASS exact saves even when stage, optimizer and checkpoint boundaries differ')
    # Exercise real HF iterable state recovery without any downloads.
    from datasets import IterableDataset
    import datasets
    datasets.config.HF_DATASETS_CACHE=root/'hf-cache'
    class Tok:
        eos_token_id=0
        def get_vocab(self): return {'last':49151}
        def encode(self,text,**kwargs): return [int(text)]*7
    metadata={'tokenizer_id':'test','tokenizer_revision':'test','dataset_id':'test','dataset_config':'test','dataset_revision':'test'}
    def make_ds(*a,**k): return IterableDataset.from_generator(lambda: ({'text':str(i%60+1)} for i in range(300)))
    with patch.object(e['AutoTokenizer'],'from_pretrained',return_value=Tok()), patch.dict(e,{'load_dataset':make_ds}):
        stream=e['TokenStream'](metadata)
        stream.batch(19,'cpu')
        state=stream.state_dict()
        expected=stream.batch(71,'cpu')
        restored=e['TokenStream'](metadata); restored.load_state_dict(state)
        actual=restored.batch(71,'cpu')
        for a,b in zip(expected,actual): torch.testing.assert_close(a,b)
    print('PASS actual HF streaming cursor and leftover-token recovery')

    # Discovery must recover the last committed checkpoint after a failed write.
    checkpoint_dir = source.parent.parent
    incomplete = checkpoint_dir / 'tokens-000000000099.incomplete-test'
    incomplete.mkdir()
    (incomplete / 'checkpoint.pt').write_bytes(b'abandoned')
    (incomplete / 'manifest.json').write_text(json.dumps({'tokens':99,'bytes':9,'resumable':True}))
    bad = checkpoint_dir / 'tokens-000000000100'
    bad.mkdir()
    (bad / 'manifest.json').write_text('{broken')
    with warnings.catch_warnings(record=True) as caught:
        assert e['latest_full'](root/'resumed') == e['checkpoint_files'](root/'resumed')[-1][1]
        assert all(i['tokens'] < 99 for i,_ in e['checkpoint_files'](root/'resumed'))
        assert caught

    def run_ui_cell(number, **controls):
        # Execute the shipped cell, substituting only UI settings and Colab storage locations.
        source_code = ''.join(n['cells'][number-1]['source'])
        source_code = '\n'.join(l for l in source_code.splitlines() if not l.startswith('%'))
        source_code = source_code.replace('/content/drive', (root/'mock-drive').as_posix())
        source_code = source_code.replace("'/content'", repr((root/'downloads').as_posix()))
        tree = ast.parse(source_code)
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                name = stmt.targets[0].id
                if name in controls:
                    stmt.value = ast.parse(repr(controls.pop(name)), mode='eval').body
        assert not controls, controls
        exec(compile(ast.fix_missing_locations(tree), f'cell_{number}', 'exec'), e)

    calls, downloads = [], []
    colab = types.ModuleType('google.colab')
    colab.drive = types.SimpleNamespace(mount=calls.append)
    colab.files = types.SimpleNamespace(download=downloads.append)
    with patch.dict(sys.modules, {'google.colab':colab}):
        run_ui_cell(7, SAVE_TO_DRIVE=True)
        drive_root=e['CHECKPOINT_ROOT']
        assert calls and drive_root.is_dir()
        e['CHECKPOINT_ROOT']=root/'baseline'
        e['RESUME_FROM']='auto'
        e['TARGET_TOKENS']=64
        original_train=e['train_gu']
        with patch.dict(e, {'ModelConfig':lambda:c,'RunConfig':lambda **kw:r,
                            'train_gu':lambda *a,**kw:original_train(*a,device='cpu',data_factory=Data,**kw)}):
            run_ui_cell(19)
            trained_model, seen = e['model'], e['tokens_seen']
        e.update(model=trained_model, tokens_seen=seen)
        assert e['tokens_seen']==64 and e['model'].source_metadata==baseline.source_metadata

        class ValData(Data):
            def __init__(self, meta, validation=False):
                assert validation and meta==baseline.source_metadata
                super().__init__(meta)
        with patch.dict(e, {'TokenStream':ValData}):
            run_ui_cell(21,RUN_VALIDATION=True)
        assert e['model'].training
        run_ui_cell(23,EXPORT_ACTION='download_model',EXPORT_TOKENS=64)
        model_export=pathlib.Path(downloads[-1])
        compact=torch.load(model_export,weights_only=True)
        assert compact['tokens']==64 and not compact['resumable'] and 'optimizer' not in compact
        for name,w in baseline.state_dict().items():
            torch.testing.assert_close(compact['weights'][name],w.bfloat16(),rtol=0,atol=0)
        run_ui_cell(23,EXPORT_ACTION='download_model',EXPORT_TOKENS=16)
        early=torch.load(downloads[-1],weights_only=True)
        assert early['tokens']==16 and early['resumable'] is False
        run_ui_cell(23,EXPORT_ACTION='download_resume',EXPORT_TOKENS=32)
        with zipfile.ZipFile(downloads[-1]) as archive:
            assert set(archive.namelist())=={'checkpoint.pt','manifest.json'} and archive.testzip() is None
            archive.extractall(root/'restored-zip')
        replay, t=original_train(c,r,root/'zip-run',root/'restored-zip'/'checkpoint.pt',device='cpu',data_factory=Data)
        assert t==64
        for a,b in zip(baseline.parameters(),replay.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        run_ui_cell(23,EXPORT_ACTION='copy_to_drive',EXPORT_TOKENS=32)
        copy_on_drive=e['latest_full'](drive_root)
        assert copy_on_drive and copy_on_drive.read_bytes()==checkpoints[1][1].read_bytes()
        class PromptTokenizer:
            eos_token_id=-1
            def encode(self,text,return_tensors): return torch.arange(20).unsqueeze(0)
            def decode(self,ids,skip_special_tokens): return str(len(ids))
        with patch.object(e['AutoTokenizer'],'from_pretrained',return_value=PromptTokenizer()) as tok:
            run_ui_cell(25,MODEL_FILE=str(model_export))
            tok.assert_called_once_with('offline-test',revision='test')
            assert e['generate_gu'](e['inference_model'],e['tokenizer'],'test',max_new_tokens=2)=='22'
        original_to = torch.nn.Module.to
        for native_bf16, expected_dtype in [(False, torch.float16), (True, torch.bfloat16)]:
            converted = []
            def simulate_gpu_to(module, *, device, dtype):
                assert device == torch.device('cuda')
                converted.append(dtype)
                return original_to(module, device='cpu', dtype=dtype)
            with patch.object(torch.cuda, 'is_bf16_supported', return_value=native_bf16) as bf16_check, \
                 patch.object(torch.nn.Module, 'to', simulate_gpu_to), \
                 patch.object(e['AutoTokenizer'], 'from_pretrained', return_value=PromptTokenizer()):
                loaded, _ = e['load_gu_inference'](model_export, torch.device('cuda'))
                assert converted == [expected_dtype]
                assert next(loaded.parameters()).dtype == expected_dtype
                bf16_check.assert_called_once_with(including_emulation=False)
        print('PASS GPU inference chooses FP16 for T4 and native BF16 where supported (simulated)')
        print('PASS notebook setup, automatic resume, held-out evaluation, both model exports, '
              'resume ZIP replay, discoverable Drive copy, and loaded-model generation (Colab UI mocked)')
finally:
    assert root.resolve().parent==pathlib.Path.cwd().resolve()
    shutil.rmtree(root)
