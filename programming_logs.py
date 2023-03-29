

'''

# 查模型大小
def main(args):

    pl.seed_everything(args.seed)


    # 模型加载路径
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


'''
# 更改主程序
    def training_epoch_end(self, outputs):
        avg_train_u_accuracy = np.stack([x[3]['train_u_accuracy'] for x in outputs]).mean()
        loss_adversarial_phi_theta_xi_stack = torch.stack([x[3]['loss_adversarial_phi_theta_xi'].detach() for x in outputs])
        loss_phi_theta_xi_stack = torch.stack([x[0]['loss_phi_theta_xi'].detach() for x in outputs])
        avg_train_loss = (loss_adversarial_phi_theta_xi_stack + loss_phi_theta_xi_stack).mean()
        tensorboard_log = {'avg_train_u_accuracy': avg_train_u_accuracy, 'avg_train_loss': avg_train_loss}

        #self.logger.experiment.add_scale('avg_train_u_accuracy', avg_train_u_accuracy)
        #self.logger.experiment.add_scale('avg_train_loss', avg_train_loss)

        self.log(name = 'training_epoch_end', value=tensorboard_log, on_epoch=True, prog_bar=True)
    
    
    
    def validation_end(self, outputs):
        print('-'*30)
        print('validation outputs', outputs)
        avg_val_u_accuracy = np.stack([x['val_u_accuracy'] for x in outputs]).mean()
        avg_val_loss = torch.stack([x['val_loss_total'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_u_accuracy': avg_val_u_accuracy, 'avg_val_loss': avg_val_loss}
        self.log_dict(tensorboard_logs, on_epoch=True, prog_bar=True)

    def test_end(self, outputs):
        avg_test_u_accuracy = np.stack([x['test_u_accuracy'] for x in outputs]).mean()
        avg_test_loss = torch.stack([x['test_loss_total'] for x in outputs]).mean()
        tensorboard_logs = {'avg_test_u_accuracy': avg_test_u_accuracy, 'avg_test_loss': avg_test_loss}
        self.log_dict(tensorboard_logs, on_epoch=True, prog_bar=True)
        
    
'''

