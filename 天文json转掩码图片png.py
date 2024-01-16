import json
import os
from PIL import Image, ImageDraw

# Create a folder to save the images
output_folder = "天文json转掩码图文件夹valid"
os.makedirs(output_folder, exist_ok=True)

# Load data from JSON file
with open('tianwen_json/val_via_region_data.json', 'r') as json_file:
    json_data = json.load(json_file)

# Iterate through all JSON objects
for image_id, image_data in json_data.items():
    # Get polygon coordinate data
    polygon_data = image_data["regions"][0]["shape_attributes"]
    x_coordinates = polygon_data["all_points_x"]
    y_coordinates = polygon_data["all_points_y"]

    # Create a blank 256x256 image with a transparent background
    width, height = 256, 256
    mask_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    # Create a drawing object
    draw = ImageDraw.Draw(mask_image)

    # Fill the polygon area with a solid color and set the background to transparent
    polygon_points = list(zip(x_coordinates, y_coordinates))
    draw.polygon(polygon_points, fill=(255, 0, 0, 255))

    # Get the file name
    filename = image_data["filename"]

    # Build the complete save path
    save_path = os.path.join(output_folder, filename)

    # Save as PNG image with transparency
    mask_image.save(save_path, "PNG")

    print(f"Mask image saved as {save_path}")
# import json
# import os
# from PIL import Image, ImageDraw
#
# # 创建保存图片的文件夹
# output_folder = "天文json转掩码图文件夹valid"
# os.makedirs(output_folder, exist_ok=True)
#
# # 从JSON文件中读取数据
# with open('tianwen_json/val_via_region_data.json', 'r') as json_file:
#     json_data = json.load(json_file)
#
# # 遍历所有JSON对象
# for image_id, image_data in json_data.items():
#     # 获取多边形的坐标数据
#     polygon_data = image_data["regions"][0]["shape_attributes"]
#     x_coordinates = polygon_data["all_points_x"]
#     y_coordinates = polygon_data["all_points_y"]
#
#     # 创建一个空白的256x256的图像
#     width, height = 256, 256
#     mask_image = Image.new("RGB", (width, height), (0, 0, 0))
#
#     # 创建一个绘图对象
#     draw = ImageDraw.Draw(mask_image)
#
#     # 填充多边形区域
#     draw.polygon(list(zip(x_coordinates, y_coordinates)), fill=(255, 0, 0))
#
#     # 获取文件名
#     filename = image_data["filename"]
#
#     # 构建完整的保存路径
#     save_path = os.path.join(output_folder, filename)
#
#     # 保存为PNG图像
#     mask_image.save(save_path)
#
#     print(f"掩码图像已保存为 {save_path}")
