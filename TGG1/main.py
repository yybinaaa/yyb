# file: main.py
import argparse
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Main entry point for the OneRec-V2 pipeline.")
    parser.add_argument('--stage', required=True, choices=['preprocess', 'sft', 'rl'],
                        help="Which stage of the pipeline to run.")
    # 我们可以把其他所有脚本的参数都集中到这里
    parser.add_argument('--data_dir', default=os.environ.get('TRAIN_DATA_PATH', '.'), type=str)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--base_model_path', default=None, type=str)
    # ... 其他所有脚本可能用到的参数 ...
    
    args, unknown = parser.parse_known_args() # 解析已知参数，忽略未知参数

    # 将所有参数转换为命令行字符串
    def build_command(script_name):
        cmd = ['python', '-u', script_name]
        # 自动将所有接收到的参数传递给子脚本
        for i in range(0, len(unknown), 2):
            cmd.append(unknown[i])
            if i + 1 < len(unknown):
                cmd.append(unknown[i+1])
        
        # 确保 data_dir 被正确传递
        cmd.extend(['--data_dir', args.data_dir])
        return cmd

    if args.stage == 'preprocess':
        print("--- Running Preprocessing Stage ---")
        # 1. Train SFT for embedding
        cmd_sft_embed = build_command('main_v2.py') + ['--mode', 'embedding', '--output_model_name', 'sft_for_embedding.pth']
        subprocess.run(cmd_sft_embed, check=True)
        
        # 2. Prepare embeddings
        cmd_prep_embed = build_command('prepare_embeddings.py') + ['--model_checkpoint', 'sft_for_embedding.pth']
        subprocess.run(cmd_prep_embed, check=True)

        # 3. Train Tokenizer
        cmd_train_tok = build_command('train_tokenizer_v2.py')
        subprocess.run(cmd_train_tok, check=True)

        # 4. Generate Semantic Map
        cmd_gen_map = build_command('generate_semantic_ids.py')
        subprocess.run(cmd_gen_map, check=True)

    elif args.stage == 'sft':
        print("--- Running SFT Stage ---")
        cmd_sft = build_command('main_v2.py') + ['--mode', 'semantic']
        subprocess.run(cmd_sft, check=True)
        
    elif args.stage == 'rl':
        print("--- Running RL Stage ---")
        if not args.base_model_path:
            raise ValueError("--base_model_path is required for RL stage.")
        cmd_rl = build_command('main_rl_v2.py') + ['--base_model_path', args.base_model_path]
        subprocess.run(cmd_rl, check=True)

if __name__ == '__main__':
    main()