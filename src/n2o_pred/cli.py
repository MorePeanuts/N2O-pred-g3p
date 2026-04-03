"""
命令行接口模块
"""

import argparse
import sys
from pathlib import Path

from .compare import compare_experiments
from .experiment import ExperimentRunner
from .predict import predict_tif_data, predict_with_model
from .preprocessing import preprocess_data
from .trainer import RFTrainConfig, RNNTrainConfig
from .utils import create_logger, load_json

logger = create_logger(__name__)


def cmd_data(args):
    """数据预处理命令"""
    if args.preprocessing:
        logger.info('开始数据预处理...')
        sequences = preprocess_data()
        logger.info(f'数据预处理完成，共 {len(sequences)} 个序列')
    else:
        logger.error('请指定 --preprocessing 参数')
        sys.exit(1)


def cmd_train(args):
    """训练命令"""
    # 确定模型类型
    model_type = args.model
    if model_type not in ['rf', 'rnn-obs', 'rnn-daily']:
        logger.error(f'不支持的模型类型: {model_type}')
        sys.exit(1)

    # 确定种子
    split_seeds = None
    if args.seed_from:
        # 从文件加载种子
        seed_from_path = Path(args.seed_from)
        if not seed_from_path.exists():
            logger.error(f'文件不存在: {seed_from_path}')
            sys.exit(1)

        summary = load_json(seed_from_path)
        split_seeds = summary['seeds']
        logger.info(f'从 {seed_from_path} 加载种子: {split_seeds}')
    elif args.seed is not None:
        # 使用单个种子
        split_seeds = [args.seed]
        logger.info(f'使用单个种子: {args.seed}')
    else:
        # 使用max_split生成多个种子
        split_seeds = None
        logger.info(f'将生成 {args.max_split} 个随机种子')

    # 创建配置
    if model_type.startswith('rnn'):
        config = RNNTrainConfig(
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            patience=args.patience,
            device=args.device,
        )
    else:
        config = RFTrainConfig(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )

    # 创建实验管理器
    output_dir = args.output if args.output else None

    if model_type.startswith('rnn'):
        runner = ExperimentRunner(
            model_type=model_type,
            output_dir=output_dir,
            device=args.device,
            rnn_config=config,
        )
    else:
        runner = ExperimentRunner(model_type=model_type, output_dir=output_dir, rf_config=config)

    # 运行实验
    if split_seeds is not None:
        summary = runner.run(
            split_seeds=split_seeds,
            train_split=args.train_split,
            max_workers=args.max_workers,
        )
    else:
        summary = runner.run(
            total_splits=args.max_split,
            split_seed=args.split_seed,
            train_split=args.train_split,
            max_workers=args.max_workers,
        )

    logger.info('训练完成！')
    logger.info(f'结果保存在: {runner.output_dir}')


def cmd_compare(args):
    """模型对比命令"""
    if len(args.models) < 2:
        logger.error('至少需要2个实验目录进行对比')
        sys.exit(1)

    # 检查目录是否存在
    for model_dir in args.models:
        if not Path(model_dir).exists():
            logger.error(f'目录不存在: {model_dir}')
            sys.exit(1)

    logger.info(f'比较 {len(args.models)} 个实验...')

    output_dir = args.output if args.output else None
    summary = compare_experiments(args.models, output_dir)

    logger.info('对比完成！')
    if summary:
        logger.info(f'最佳模型(按R2): {summary["best_model_by_R2"]["experiment"]}')
        logger.info(f'  R2: {summary["best_model_by_R2"]["test_R2"]:.4f}')


def cmd_predict(args):
    """预测命令"""
    model_dir = Path(args.model)
    if not model_dir.exists():
        logger.error(f'模型目录不存在: {model_dir}')
        sys.exit(1)

    data_path = Path(args.dataset)
    if not data_path.exists():
        logger.error(f'数据路径不存在: {data_path}')
        sys.exit(1)

    # 解析plot参数
    plot_sequences = None
    if args.plot:
        plot_sequences = []
        for plot_arg in args.plot:
            parts = plot_arg.split(',')
            if len(parts) != 2:
                logger.error(f'无效的序列格式: {plot_arg}，应为 "publication,control_group"')
                sys.exit(1)
            plot_sequences.append((parts[0], parts[1]))
        logger.info(f'将绘制 {len(plot_sequences)} 个序列的预测图')

    # 检查是否为TIF目录（包含TIF文件的目录）
    is_tif_dir = data_path.is_dir() and any(data_path.glob('*.tif'))

    if is_tif_dir:
        # TIF格式数据预测
        output_dir = args.output if args.output else f'predictions_{data_path.name}'

        logger.info(f'检测到TIF目录，使用模型 {model_dir} 对 {data_path} 进行预测...')

        results = predict_tif_data(
            model_dir=model_dir,
            tif_dir=data_path,
            output_dir=output_dir,
            device=args.device,
            batch_size=args.batch_size,
        )

        logger.info('预测完成！')
        logger.info(f'生成文件数: {len(results["completed_files"])}')
        logger.info(f'总处理像素数: {results["total_pixels_processed"]}')
        logger.info(f'输出目录: {results["output_dir"]}')
    else:
        # 常规数据预测
        output_path = args.output if args.output else None

        logger.info(f'使用模型 {model_dir} 进行预测...')

        results = predict_with_model(
            model_dir=model_dir,
            data_path=data_path,
            output_path=output_path,
            device=args.device,
            plot_sequences=plot_sequences,
        )

        logger.info('预测完成！')


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description='N2O排放预测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 数据预处理
  n2o-pred data --preprocessing
  
  # 训练随机森林模型
  n2o-pred train --model rf --seed 42
  
  # 训练RNN模型（观测步长）
  n2o-pred train --model rnn-obs --max-split 20
  
  # 训练RNN模型（每日步长）
  n2o-pred train --model rnn-daily --seed-from outputs/exp_xxx/summary.json
  
  # 比较模型
  n2o-pred compare --models outputs/exp_1 outputs/exp_2 outputs/exp_3

  # 预测（常规数据）
  n2o-pred predict --model outputs/exp_xxx/split_42 --dataset datasets/data_EUR_processed.pkl

  # 预测并绘制特定序列的预测图
  n2o-pred predict --model outputs/exp_xxx/split_42 --dataset datasets/data_EUR_processed.pkl --plot 44,3 58,3

  # 预测（TIF格式数据）
  n2o-pred predict --model outputs/exp_xxx/split_42 --dataset input_2020 --output predictions/
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # ===== data 命令 =====
    parser_data = subparsers.add_parser('data', help='数据预处理')
    parser_data.add_argument('--preprocessing', action='store_true', help='执行数据预处理')
    parser_data.set_defaults(func=cmd_data)

    # ===== train 命令 =====
    parser_train = subparsers.add_parser('train', help='训练模型')
    parser_train.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['rf', 'rnn-obs', 'rnn-daily'],
        help='模型类型',
    )

    # 种子相关参数
    seed_group = parser_train.add_mutually_exclusive_group()
    seed_group.add_argument('--seed', type=int, help='使用单个随机种子')
    seed_group.add_argument('--max-split', type=int, default=1, help='随机种子数量（默认1）')
    parser_train.add_argument('--seed-from', type=str, help='从summary.json加载种子')
    parser_train.add_argument(
        '--split-seed', type=int, default=42, help='生成随机种子的种子（默认42）'
    )

    # 训练参数
    parser_train.add_argument(
        '--train-split', type=float, default=0.8, help='训练集比例（默认0.8）'
    )
    parser_train.add_argument('--device', type=str, default='cuda:0', help='设备（默认cuda:0）')
    parser_train.add_argument(
        '--max-workers', type=int, default=1, help='最大并行训练数（默认1，串行）'
    )
    parser_train.add_argument('--output', type=str, help='输出目录')

    # RNN参数
    parser_train.add_argument('--max-epochs', type=int, default=300, help='最大训练轮次（默认300）')
    parser_train.add_argument('--batch-size', type=int, default=16, help='批次大小（默认16）')
    parser_train.add_argument('--lr', type=float, default=1e-3, help='学习率（默认0.001）')
    parser_train.add_argument('--patience', type=int, default=30, help='早停耐心值（默认30）')

    # RF参数
    parser_train.add_argument(
        '--n-estimators', type=int, default=100, help='随机森林树的数量（默认100）'
    )
    parser_train.add_argument('--max-depth', type=int, help='随机森林最大深度')

    parser_train.set_defaults(func=cmd_train)

    # ===== compare 命令 =====
    parser_compare = subparsers.add_parser('compare', help='比较模型')
    parser_compare.add_argument('--models', nargs='+', required=True, help='实验目录列表')
    parser_compare.add_argument('--output', type=str, help='输出目录')
    parser_compare.set_defaults(func=cmd_compare)

    # ===== predict 命令 =====
    parser_predict = subparsers.add_parser('predict', help='预测')
    parser_predict.add_argument('--model', type=str, required=True, help='模型目录')
    parser_predict.add_argument(
        '--dataset', type=str, required=True, help='数据路径（文件或TIF目录）'
    )
    parser_predict.add_argument('--output', type=str, help='输出路径或目录')
    parser_predict.add_argument('--device', type=str, default='cuda:0', help='设备（默认cuda:0）')
    parser_predict.add_argument(
        '--batch-size', type=int, default=256, help='批次大小（TIF预测用，默认256）'
    )
    parser_predict.add_argument(
        '--plot',
        type=str,
        nargs='*',
        help='需要绘制预测图的序列，格式为 "publication,control_group"，例如 --plot 44,3 58,3',
    )
    parser_predict.set_defaults(func=cmd_predict)

    # 解析参数
    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)

    # 执行命令
    try:
        args.func(args)
    except Exception as e:
        logger.error(f'执行失败: {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
