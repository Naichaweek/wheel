import re

# 读取txt文件内容
file_path = r'C:\Users\hpc\Desktop\fsdownload\gauss_25\gauss_25.txt'
with open(file_path, 'r') as file:
    content = file.read()

# 使用正则表达式匹配每条文本中的数值
pattern = r"Average PSNR \| noisy:(\d+\.\d+), output:(\d+\.\d+), prediction:(\d+\.\d+)"
matches = re.findall(pattern, content)

noisy_values = []
output_values = []
prediction_values = []

for match in matches:
    noisy_value = float(match[0])
    output_value = float(match[1])
    prediction_value = float(match[2])

    noisy_values.append(noisy_value)
    output_values.append(output_value)
    prediction_values.append(prediction_value)

# 找出每个列表中的最大值
max_noisy = max(noisy_values)
max_output = max(output_values)
max_prediction = max(prediction_values)

print("Max Noisy Value:", max_noisy)
print("Max Output Value:", max_output)
print("Max Prediction Value:", max_prediction)