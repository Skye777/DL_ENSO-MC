import argparse


class Hparams:
    parser = argparse.ArgumentParser()

    # prepro
    # parser.add_argument('--reanalysis_dataset_dir', default='/home/dl/Public/Skye/transformer/data/reanalysis_data/')
    # parser.add_argument('--observe_dataset_dir', default='/home/dl/Public/Skye/transformer/data/observe_data/')
    # parser.add_argument('--reanalysis_npz_dir',
    #                     default='/home/dl/Public/Skye/transformer/data/reanalysis_data/npz-data')
    # parser.add_argument('--observe_npz_dir',
    #                     default='/home/dl/Public/Skye/transformer/data/observe_data/npz-data')
    parser.add_argument('--soda_dataset_dir', default='D:/Python/transformer/data/soda_data/')
    parser.add_argument('--godas_dataset_dir', default='D:/Python/transformer/data/godas_data/')
    parser.add_argument('--reanalysis_dataset_dir', default='D:/Python/transformer/data/reanalysis_data/')
    parser.add_argument('--observe_dataset_dir', default='D:/Python/transformer/data/observe_data/')
    parser.add_argument('--observe_npz_dir',
                        default='D:/Python/transformer/data/observe_data/npz-data')
    parser.add_argument('--reanalysis_npz_dir',
                        default='D:/Python/transformer/data/reanalysis_data/npz-data')
    parser.add_argument('--reanalysis_preprocess_out_dir',
                        default='/home/dl/Public/Skye/transformer/data/reanalysis_data/tfRecords')
    parser.add_argument('--observe_preprocess_out_dir',
                        default='/home/dl/Public/Skye/transformer/data/observe_data/tfRecords/convlstm')

    # data
    parser.add_argument('--in_seqlen', default=12)
    parser.add_argument('--out_seqlen', default=12)
    parser.add_argument('--lead_time', default=0)
    parser.add_argument('--rolling_len', default=6)
    parser.add_argument('--width', default=160)
    parser.add_argument('--height', default=80)
    parser.add_argument('--num_predictor', default=4)
    parser.add_argument('--input_variables', default=["sst", "t300", "taux", "tauy"])
    parser.add_argument('--num_output', default=1)
    parser.add_argument('--output_variables', default=["sst"])

    # training scheme
    parser.add_argument('--strategy', default='DMS')
    parser.add_argument('--train_eval_split', default=0.2)
    parser.add_argument('--random_seed', default=2021)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--eval_batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_epoch_record', default=1, help="Number of step to record checkpoint.")

    parser.add_argument('--ckpt', default='', help="checkpoint file path")
    parser.add_argument('--single_gpu_model_dir', default="ckpt/checkpoints_single")
    parser.add_argument('--multi_gpu_model_dir', default="ckpt/checkpoints_multi")
    parser.add_argument('--lr', default=0.0003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="logs", help="log directory")

    # model
    parser.add_argument('--model', default='uconvlstm')
    parser.add_argument('--model_structure', default="Joint")
    parser.add_argument('--vunits', default=128)
    parser.add_argument('--Tunits', default=8)
    parser.add_argument('--Munits', default=8)
    parser.add_argument('--MTunits', default=8)
    parser.add_argument('--V_kernel', default=3)
    parser.add_argument('--V_stride', default=1)
    parser.add_argument('--d_model', default=1024, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen1', default=100, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=100, type=int,
                        help="maximum length of a target sequence")
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

    # test scheme
    parser.add_argument('--test_month_start', default=1)
    parser.add_argument('--delivery_model_dir', default='D:/Python/transformer/delivery/test/sst+t300+ssh+wind')
    parser.add_argument('--delivery_model_file', default='uconvlstm-ckp_89')
    parser.add_argument('--explain_model_file', default='uconvlstm-sst+t300+wind_ckp_80')
    # testRolling
    parser.add_argument('--delivery_sst_model_file', default='uconvlstm-lead0_ckp_65')
    parser.add_argument('--delivery_t300_model_file', default='t300/uconvlstm-t300+uw_ckp_78')
    parser.add_argument('--delivery_uw_model_file', default='uw/uconvlstm-uw_ckp_22')
    parser.add_argument('--delivery_vw_model_file', default='vw/uconvlstm-vw+sst_ckp_29')

    # predictor test
    parser.add_argument('--sst', default='D:/Python/transformer/predictorTest/sst')
    parser.add_argument('--ssta', default='D:/Python/transformer/predictorTest/ssta')
    parser.add_argument('--multivar', default='D:/Python/transformer/predictorTest/multivar')
    parser.add_argument('--directRollingPred', default='D:/Python/transformer/directRollingPred')
    parser.add_argument('--rollingPredict', default='D:/Python/transformer/rollingPredict')
    parser.add_argument('--explain', default='D:/Python/transformer/explain')
    parser.add_argument('--result', default='D:/Python/transformer/result')
    parser.add_argument('--saliency_npz', default='D:/Python/transformer/result/saliency_npz/uconvlstm-ckp_89/before_peak/cold')
    parser.add_argument('--sensitiveExp', default='D:/Python/transformer/sensitiveExp')
