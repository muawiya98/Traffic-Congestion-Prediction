# -- coding: utf-8 --

import argparse
import os
root_path = 'E:\\Tasks\\09. Traffic Congestion Prediction\\STGIN-main'
# Chang it for each parameter ('Results_KNN', 'Results_K', Results_R)
main_results_path = os.path.join(root_path, 'Results_KNN')  
data_path = "E:\\Tasks\\09. Traffic Congestion Prediction\\Traffic Congestion Prediction\\Dataset"
class parameter(object):
    def __init__(self, parser, Horizon=12, r_ation=0, data_name='METRLA', method_name='STGIN', knn=0.9):
        if not os.path.exists(main_results_path): os.makedirs(main_results_path)
        save_path = os.path.join(main_results_path, 'weights_'+data_name+'_'+method_name+'_Horizon'+str(Horizon)+'KNN_'+str(knn))
        if not os.path.exists(save_path): os.makedirs(save_path)
        save_path = os.path.join(save_path, 'weights_'+data_name+'_'+method_name+'_Horizon'+str(Horizon)+'KNN_'+str(knn))
        results_path = os.path.join(main_results_path, 'Results_'+data_name+'_'+method_name+'_Horizon'+str(Horizon)+'KNN_'+str(knn)+'_'+str(r_ation))
        if not os.path.exists(results_path): os.makedirs(results_path)
        file_adj = os.path.join(data_path, data_name, 'adjacent_'+str(knn)+'.csv')
        
        if data_name == 'METRLA':
            file_train_s = os.path.join(data_path, data_name, 'M_Metrla_New.csv')
            site_num = 207

        elif data_name == 'PEMSBAY':            
            file_train_s =  os.path.join(data_path, data_name, 'M_Pemsbay_New.csv')
            site_num = 325


        self.parser=parser
        self.parser.add_argument('--save_path', type=str, default=save_path, help='save path')
        self.parser.add_argument('--results_path', type=str, default=results_path, help='results path')
        self.parser.add_argument('--model_name', type=str, default=method_name, help='training or testing model name')
        self.parser.add_argument('--file_train_s', type=str, default=file_train_s, help='training_speed file address')
        self.parser.add_argument('--file_adj', type=str,default=file_adj, help='adj file address')
        self.parser.add_argument('--site_num', type=int, default=site_num, help='total number of road') ####
        self.parser.add_argument('--granularity', type=int, default=5, help='minute granularity') ####
        self.parser.add_argument('--train_ratio', type=float, default=0.7, help='train data divide')
        self.parser.add_argument('--validate_ratio', type=float, default=0.1, help='validate divide')
        self.parser.add_argument('--test_ratio', type=float, default=0.2, help='test divide')
        self.parser.add_argument('--is_training', type=bool, default=True, help='is training')
        self.parser.add_argument('--epoch', type=int, default=10, help='epoch') # 100
        self.parser.add_argument('--step', type=int, default=1, help='step')
        self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--dropout', type=float, default=0.0, help='drop out')
        self.parser.add_argument('--decay_epoch', type=int, default=5, help='decay epoch')
        self.parser.add_argument('--num_heads', type=int, default=8, help='total number of head attentions')
        self.parser.add_argument('--num_blocks', type=int, default=1, help='total number of attention layers')

        #每个点表示a->b路线，目前8个收费站
        self.parser.add_argument('--emb_size', type=int, default=64, help='embedding size')
        self.parser.add_argument('--features', type=int, default=1, help='numbers of the feature')
        self.parser.add_argument('--features_p', type=int, default=15, help='numbers of the feature pollution')
        self.parser.add_argument('--normalize', type=bool, default=True, help='normalize')
        self.parser.add_argument('--input_length', type=int, default=Horizon, help='input length')
        self.parser.add_argument('--output_length', type=int, default=Horizon, help='output length')
        self.parser.add_argument('--r_ratio', type=int, default=r_ation, help='R Ratio')


        self.parser.add_argument('--hidden1', type=int, default=32, help='number of units in hidden layer 1')
        self.parser.add_argument('--gcn_output_size', type=int, default=64, help='model string')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight for L2 loss on embedding matrix')
        self.parser.add_argument('--max_degree', type=int, default=3, help='maximum Chebyshev polynomial degree')

        self.parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
        self.parser.add_argument('--hidden_layer', type=int, default=1, help='hidden layer')

        self.parser.add_argument('--training_set_rate', type=float, default=0.7, help='training set rate')
        self.parser.add_argument('--validate_set_rate', type=float, default=0.1, help='validate set rate')
        self.parser.add_argument('--test_set_rate', type=float, default=0.20, help='test set rate')


    def get_para(self):
        return self.parser.parse_args()

if __name__=='__main__':
    para=parameter(argparse.ArgumentParser())

    print(para.get_para().batch_size)
