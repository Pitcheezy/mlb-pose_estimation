# 필요한 도구들을 불러옵니다.
import pandas as pd
from pybaseball import statcast_single_game

# --- 1. 요청하신 2025년 미래 경기의 ID ---
# 이 경기는 미래의 경기이므로, 실제 경기가 끝나기 전까지는 데이터를 가져올 수 없습니다.
future_game_pk = 813024

print(f"요청하신 미래 경기(ID: {future_game_pk})의 데이터 추출을 시작합니다.")
print("경기가 아직 열리지 않았으므로, 데이터가 비어있을 수 있습니다.")

try:
    # 특정 경기의 모든 투구 데이터를 가져오는 함수를 실행합니다.
    future_game_data = statcast_single_game(future_game_pk)

    # 만약 데이터를 성공적으로 가져왔다면
    if future_game_data is not None and not future_game_data.empty:
        print(f"\n--- 경기 ID: {future_game_pk} 데이터 결과 (일부) ---")
        print(future_game_data.head())
        # 가져온 데이터를 CSV 파일로 저장합니다.
        future_csv_filename = f'mlb_game_{future_game_pk}_data.csv'
        future_game_data.to_csv(future_csv_filename, index=False, encoding='utf-8-sig')
        print(f"\n성공! {future_csv_filename} 이름으로 데이터가 파일로 저장되었습니다.")
    # 데이터를 가져오지 못했다면
    else:
        print(f"경기 ID {future_game_pk}에 대한 데이터를 찾을 수 없습니다. 아직 진행되지 않은 경기입니다.")

except Exception as e:
    print(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")

# 보기 좋게 구분선을 넣습니다.
print("\n" + "="*50 + "\n")

# --- 2. 코드 실행 확인을 위한 과거 경기 예시 ---
# 2023년 월드시리즈 5차전 경기의 ID는 748571입니다.
past_game_pk = 748571

print(f"코드 실행 테스트를 위해, 실제 진행됐던 과거 경기(ID: {past_game_pk}) 데이터를 가져옵니다.")

try:
    # 과거 경기의 데이터를 가져옵니다.
    past_game_data = statcast_single_game(past_game_pk)
    # 데이터를 성공적으로 가져왔는지 확인합니다.
    if past_game_data is not None and not past_game_data.empty:
        # 모든 열이 다 보이도록 설정합니다.
        pd.set_option('display.max_columns', None)

        print(f"\n--- 경기 ID: {past_game_pk} 데이터 결과 (일부) ---")
        # 가져온 데이터의 앞부분 5줄을 화면에 보여줍니다.
        print(past_game_data.head())

        # 가져온 데이터를 엑셀과 비슷한 CSV 파일로 저장합니다.
        csv_filename = f'mlb_game_{past_game_pk}_data.csv'
        past_game_data.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        print(f"\n성공! {csv_filename} 이름으로 데이터가 파일로 저장되었습니다.")
    else:
        print(f"경기 ID {past_game_pk}에 대한 데이터를 가져오지 못했습니다. 일시적인 네트워크 문제일 수 있습니다.")

except Exception as e:
    print(f"데이터를 가져오는 중 오류가 발생했습니다: {e}")