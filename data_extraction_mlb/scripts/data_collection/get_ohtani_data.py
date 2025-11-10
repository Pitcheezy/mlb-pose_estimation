# get_ohtani_data.py 의 전체 내용
import pandas as pd
from pybaseball import playerid_lookup, statcast_pitcher
from datetime import datetime

print("오타니 쇼헤이 선수의 MLB ID를 찾습니다...")
try:
    player_info = playerid_lookup('ohtani', 'shohei')
    if player_info is None or player_info.empty:
        print("오류: 오타니 쇼헤이 선수의 ID를 찾을 수 없습니다. 철자를 확인해주세요.")
        exit()
    ohtani_id = int(player_info['key_mlbam'].iloc[0])
    print(f"성공! 오타니 쇼헤이의 선수 ID는 '{ohtani_id}' 입니다.")

    start_date = '2018-03-01'
    end_date = datetime.today().strftime('%Y-%m-%d')

    print(f"\n{start_date}부터 {end_date}까지의 모든 투구 데이터를 수집합니다.")
    print("데이터 양이 많아 몇 분 정도 소요될 수 있습니다. 잠시만 기다려주세요...")

    ohtani_data = statcast_pitcher(start_dt=start_date, end_dt=end_date, player_id=ohtani_id)

    if ohtani_data is not None and not ohtani_data.empty:
        print("\n데이터 수집 완료! 가져온 데이터의 일부는 다음과 같습니다.")
        print(ohtani_data.head())
        total_pitches = len(ohtani_data)
        print(f"\n총 {total_pitches}개의 투구 데이터를 찾았습니다.")

        csv_filename = f'shohei_ohtani_pitching_data_{start_date}_to_{end_date}.csv'
        ohtani_data.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        print(f"\n★★★★★ 성공! '{csv_filename}' 이름으로 모든 데이터가 파일로 저장되었습니다. ★★★★★")
    else:
        print("오타니 쇼헤이의 투구 데이터를 해당 기간 동안 찾을 수 없었습니다.")
except Exception as e:
    print(f"\n데이터를 가져오는 중 오류가 발생했습니다: {e}")