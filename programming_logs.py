

'''

# 查模型大小
def main(args):

    pl.seed_everything(args.seed)


    # 模型加载路径，TODO：正在考察数据集模型接口
    load_path = load_model_path_by_args(args) # 测试完成 TODO:会不会需要额外写预训练的路径
    data_module = CelebaInterface(**vars(args)) # 测试完成

    if load_path is None:
        bottlenecknets = ConstructBottleneckNets(args)# 测试完成
    else:
        bottlenecknets = ConstructBottleneckNets(args)# 测试完成
        args.ckpt_path = load_path

    data_module.setup(stage='fit')
    for i, item in enumerate(data_module.train_dataloader()):
        x, u, s = item
        features = x
        break

    model_size = pl.utilities.memory.get_model_size_mb(bottlenecknets)
    print('model_size = {}'.format(model_size))
    bottlenecknets.example_input_array = [features]
    summary = pl.utilities.model_summary.ModelSummary(bottlenecknets, max_depth = -1)
    print(summary)
'''




