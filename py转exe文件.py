import subprocess

script_path = r"C:\Users\Administrator\Desktop\一键多启动.py"
output_dir = r"C:\Users\Administrator\Desktop"

def convert_to_exe(script_path, output_dir):
    try:
        subprocess.call(["pyinstaller", "--onefile", "--noconsole", script_path, "--distpath", output_dir])
        print("成功将脚本转换为可执行文件")
    except Exception as e:
        print(f"转换为可执行文件时发生错误：{e}")

if __name__ == "__main__":
    convert_to_exe(script_path, output_dir)