import pandas as pd
import glob
import os
import time
import random
import requests
from tqdm import tqdm
from datetime import datetime

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- 1. CSV 파일 불러오기 ---
print("오타니 쇼헤이 투구 데이터 파일을 찾습니다...")
try:
    search_pattern = 'shohei_ohtani_pitching_data_*.csv'
    file_list = glob.glob(search_pattern)
    if not file_list:
        print(f"오류: '{search_pattern}' 패턴의 CSV 파일을 찾을 수 없습니다.")
        exit()
    
    latest_file = max(file_list, key=os.path.getctime)
    print(f"데이터 파일을 찾았습니다: '{latest_file}'")
    df = pd.read_csv(latest_file)
    print(f"총 {len(df)}개의 투구 데이터를 불러왔습니다.")

except Exception as e:
    print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
    exit()

# --- 2. 연도별 폴더 구조 생성 ---
base_folder = "ohtani_videos"
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
    print(f"기본 폴더 '{base_folder}'를 생성했습니다.")

# 연도별 폴더 생성
years = df['game_year'].unique()
for year in years:
    year_folder = f"{base_folder}/{int(year)}"
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)
        print(f"연도별 폴더 '{year_folder}'를 생성했습니다.")

print(f"총 {len(years)}개 연도의 폴더를 생성했습니다.")

# --- 3. Baseball Savant 검색 페이지에서 비디오 다운로드 ---
print("\nBaseball Savant 검색 페이지에서 비디오를 찾습니다...")

# 웹 드라이버 설정
    options = webdriver.ChromeOptions()
options.add_argument('--headless=new')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('--window-size=1920,1080')
options.add_argument('--disable-gpu')
options.add_argument('--log-level=3')
options.add_argument('--disable-extensions')
options.add_argument('--disable-plugins')
options.add_argument('--disable-images')
options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 30)

downloaded_count = 0
skipped_count = 0
error_count = 0

# 연도별로 처리
for year in sorted(years, reverse=True):  # 최신 연도부터
    print(f"\n=== {year}년 데이터 처리 시작 ===")

    year_df = df[df['game_year'] == year]
    year_folder = f"{base_folder}/{int(year)}"

    # 연도별 검색 URL 생성
    search_url = f"https://baseballsavant.mlb.com/statcast_search?hfPT=&hfAB=&hfGT=R%7C&hfPR=&hfZ=&hfStadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea={int(year)}%7C&hfSit=&player_type=pitcher&hfOuts=&hfOpponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfMo=&hfTeam=&home_road=&hfRO=&position=&hfInfield=&hfOutfield=&hfInn=&hfBBT=&hfFlag=&pitchers_lookup%5B%5D=660271&metric_1=&group_by=name&min_pitches=0&min_results=0&min_pas=0&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=desc#results"

    try:
        # 검색 페이지 방문
        print(f"{year}년 검색 페이지 방문...")
        driver.get(search_url)
        time.sleep(5)

        # 검색 결과 테이블 찾기
        try:
            result_table = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table#search_results")))
        except:
            print(f"{year}년: 검색 결과 테이블을 찾을 수 없습니다. 건너뜁니다.")
            continue

        # tbody에서 행 찾기
        tbody = result_table.find_element(By.TAG_NAME, "tbody")
        rows = tbody.find_elements(By.TAG_NAME, "tr")

        if not rows:
            print(f"{year}년: 검색 결과가 없습니다.")
            continue

        print(f"{year}년: {len(rows)}개의 검색 결과를 찾았습니다.")

        # 첫 번째 행(오타니) 클릭해서 비디오 링크들 표시
        first_row = rows[0]
        print(f"{year}년: 첫 번째 행을 클릭하여 비디오 링크들을 표시합니다...")

        first_row.click()
        time.sleep(3)

        # 나타난 비디오 링크들 수집
        video_elements = driver.find_elements(By.CSS_SELECTOR, "a[href*='/sporty-videos?playId=']")
        print(f"{year}년: 총 {len(video_elements)}개의 비디오 링크를 찾았습니다.")

        # 각 비디오 링크 처리 (전체 다운로드)
        max_downloads = len(video_elements)
        print(f"{year}년: {max_downloads}개 비디오를 다운로드합니다.")

        for i, video_elem in enumerate(video_elements[:max_downloads]):
            try:
                video_url = video_elem.get_attribute('href')
                if not video_url:
            continue

                print(f"\n{year}년 - 비디오 {i+1}/{max_downloads} 다운로드 중...")

                # 새 탭에서 열기
                driver.execute_script("window.open('');")
                driver.switch_to.window(driver.window_handles[-1])
                driver.get(video_url)
                time.sleep(2)

                # 비디오 소스 찾기
                video_src = None
                try:
                    source_tag = driver.find_element(By.XPATH, "//video/source")
                    video_src = source_tag.get_attribute('src')
                except:
                    try:
                        video_tag = driver.find_element(By.TAG_NAME, "video")
                        video_src = video_tag.get_attribute('src')
                    except:
                        pass

                # CSV에서 해당 비디오의 메타데이터 찾기
                play_id = video_url.split('playId=')[-1]
                # 실제로는 playId로 CSV 매칭이 어려움, 인덱스로 대략 매칭
                # (실제 운영에서는 더 정확한 매칭 로직 필요)

                # 현재 비디오의 순서로 CSV 데이터 추정 (근사치)
                if i < len(year_df):
                    row = year_df.iloc[i]
                    game_date = str(row['game_date']).split(' ')[0] if pd.notna(row['game_date']) else 'unknown'
                    game_pk = int(row['game_pk']) if pd.notna(row['game_pk']) else 0
                    at_bat_num = int(row['at_bat_number']) if pd.notna(row['at_bat_number']) else 0
                    pitch_num = int(row['pitch_number']) if pd.notna(row['pitch_number']) else 0
                    pitch_type = str(row['pitch_type']) if pd.notna(row['pitch_type']) else 'UNK'
                    pitch_name = str(row['pitch_name']).replace(' ', '_') if pd.notna(row['pitch_name']) else 'Unknown'
                    events = str(row['events']).replace(' ', '_') if pd.notna(row['events']) else 'none'

                    # 파일명 생성 (러닝에 적합한 구조)
                    filename = f"{year_folder}/{game_date}_{game_pk}_atbat_{at_bat_num}_pitch_{pitch_num}_{pitch_type}_{pitch_name}_{events}.mp4"
                else:
                    # CSV 데이터가 부족한 경우 기본 파일명
                    filename = f"{year_folder}/{year}_video_{i+1}.mp4"

                # 이미 다운로드된 파일이면 건너뛰기
                if os.path.exists(filename):
                    print(f"이미 존재: {os.path.basename(filename)}")
                    skipped_count += 1
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    continue

                # 비디오 다운로드
                if video_src and video_src.startswith('http') and 'm3u8' not in video_src:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Referer': video_url,
                        'Accept': '*/*'
                    }

                    try:
                        response = requests.get(video_src, stream=True, headers=headers, timeout=60)
            if response.status_code == 200:
                            with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                        f.write(chunk)
                            downloaded_count += 1
                            print(f"성공: {os.path.basename(filename)}")
            else:
                            print(f"다운로드 실패: HTTP {response.status_code}")
                            error_count += 1
                    except Exception as e:
                        print(f"다운로드 오류: {e}")
                        error_count += 1
        else:
                    print("유효한 비디오 URL을 찾지 못했습니다.")
                    error_count += 1

                # 탭 닫기
                driver.close()
                driver.switch_to.window(driver.window_handles[0])

                # 서버 부하 방지를 위한 대기
                time.sleep(random.uniform(1, 3))

            except Exception as e:
                print(f"비디오 처리 오류: {e}")
                error_count += 1
                # 탭 정리
                if len(driver.window_handles) > 1:
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                continue

    except Exception as e:
        print(f"{year}년 처리 중 오류: {e}")
        continue

# --- 5. 작업 종료 ---
driver.quit()

print(f"\n{'='*50}")
print("모든 작업이 완료되었습니다!")
print(f"다운로드 성공: {downloaded_count}개")
print(f"건너뜀: {skipped_count}개")
print(f"오류: {error_count}개")
print(f"총 처리: {downloaded_count + skipped_count + error_count}개")
print(f"\n저장 위치: {base_folder}/")
print("\n폴더 구조:")
print("- ohtani_videos/")
print("  ├── 2025/")
print("  │   ├── 2025-11-01_813024_atbat_1_pitch_1_FF_4-Seam_Fastball_foul.mp4")
print("  │   └── ...")
print("  ├── 2024/")
print("  └── ...")