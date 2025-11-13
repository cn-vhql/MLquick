import sys
import os
import subprocess

def main():
    """
    MLquick 启动入口
    启动 Streamlit 应用程序
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建 MLquick.py 的完整路径
    mlquick_path = os.path.join(current_dir, "src", "MLquick.py")
    
    # 检查文件是否存在
    if not os.path.exists(mlquick_path):
        print(f"错误: 找不到文件 {mlquick_path}")
        return 1
    
    print("正在启动 MLquick 机器学习平台...")
    print(f"应用路径: {mlquick_path}")
    print("请等待浏览器自动打开...")
    
    try:
        # 使用 subprocess 启动 streamlit
        # 直接运行 streamlit run 命令
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", mlquick_path
        ], check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"启动失败: {e}")
        return 1
    except FileNotFoundError:
        print("错误: 未找到 streamlit，请确保已安装 streamlit")
        print("安装命令: pip install streamlit")
        return 1
    except KeyboardInterrupt:
        print("\n应用已停止")
        return 0
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
