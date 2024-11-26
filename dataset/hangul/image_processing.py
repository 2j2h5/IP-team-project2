import os

# 대상 폴더 경로
folder_path = "images"

# 폴더에서 파일 이름을 가져오기
files = [f for f in os.listdir(folder_path) if f.startswith("hangul_") and f.endswith(".jpeg")]

# 파일 이름에서 번호를 추출하고 정렬
file_numbers = sorted(int(f.split('_')[1].split('.')[0]) for f in files)

# 재정렬 및 파일 이름 변경
for new_index, old_number in enumerate(file_numbers, start=1):
    old_name = f"hangul_{old_number}.jpeg"
    new_name = f"hangul_{new_index}.jpeg"
    
    # 파일 이름 변경
    os.rename(os.path.join(folder_path, old_name), os.path.join(folder_path, new_name))

print("파일 이름 재정렬 완료!")
