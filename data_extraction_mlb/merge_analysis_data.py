import pandas as pd
import glob
import os

def merge_analysis_data():
    """
    원본 Statcast 데이터와 영상 분석 데이터를 병합하는 스크립트
    """

    print("🔄 오타니 투구 데이터 병합을 시작합니다...")
    print("=" * 60)

    # --- 1. 원본 Statcast 데이터 로드 ---

    # data/raw/csv 폴더에서 오타니 Statcast 원본 CSV 파일 찾기
    statcast_files = glob.glob("data/raw/csv/shohei_ohtani_pitching_data_*.csv")

    if not statcast_files:
        print("❌ 오류: 오타니 Statcast 원본 CSV 파일을 찾을 수 없습니다.")
        print("   현재 폴더에서 'shohei_ohtani_pitching_data_*.csv' 패턴의 파일을 찾았습니다.")
        return None

    # 가장 최신 파일 선택
    latest_statcast_file = max(statcast_files, key=os.path.getctime)
    print(f"📄 원본 Statcast 데이터 로드: {latest_statcast_file}")

    try:
        main_df = pd.read_csv(latest_statcast_file)
        print(f"   ✅ 로드 완료: {len(main_df)} 행")
    except Exception as e:
        print(f"❌ 오류: 원본 CSV 파일 로드 실패: {e}")
        return None

    # --- 2. 영상 분석 데이터 로드 ---

    analysis_file = "results/video_analysis_results.csv"

    if not os.path.exists(analysis_file):
        print(f"❌ 오류: 영상 분석 데이터 파일을 찾을 수 없습니다: {analysis_file}")
        return None

    print(f"📄 영상 분석 데이터 로드: {analysis_file}")

    try:
        analysis_df = pd.read_csv(analysis_file)
        print(f"   ✅ 로드 완료: {len(analysis_df)} 행")
    except Exception as e:
        print(f"❌ 오류: 영상 분석 CSV 파일 로드 실패: {e}")
        return None

    # --- 3. 데이터 병합 ---

    print("\n🔗 데이터 병합을 시작합니다...")

    # 병합 키 컬럼들을 정수 타입으로 변환
    merge_keys = ['game_pk', 'at_bat_number', 'pitch_number']

    for key in merge_keys:
        if key in main_df.columns:
            main_df[key] = main_df[key].astype(int)
        if key in analysis_df.columns:
            analysis_df[key] = analysis_df[key].astype(int)

    # 데이터 병합 (inner join)
    try:
        final_df = pd.merge(main_df, analysis_df, on=merge_keys, how='inner')
        print("   ✅ 병합 완료")
    except Exception as e:
        print(f"❌ 오류: 데이터 병합 실패: {e}")
        return None

    # --- 4. 결과 출력 ---

    print("\n📊 데이터 병합 결과:")
    print(f"   원본 Statcast 데이터: {len(main_df)} 행")
    print(f"   영상 분석 데이터: {len(analysis_df)} 행")
    print(f"   최종 병합 데이터: {len(final_df)} 행")

    # arm_angle vs calculated_release_angle 비교
    print("\n🔍 Statcast arm_angle vs 계산된 release_angle 비교:")
    if 'arm_angle' in final_df.columns and 'calculated_release_angle' in final_df.columns:
        comparison_df = final_df[['game_pk', 'at_bat_number', 'pitch_number',
                                  'arm_angle', 'calculated_release_angle']].head()
        print(comparison_df.to_string(index=False))
    else:
        print("   ⚠️  비교할 컬럼이 없습니다.")
        if 'arm_angle' not in final_df.columns:
            print("     - 'arm_angle' 컬럼이 없습니다.")
        if 'calculated_release_angle' not in final_df.columns:
            print("     - 'calculated_release_angle' 컬럼이 없습니다.")

    # --- 5. 최종 저장 ---

    output_file = "results/FINAL_ohtani_data_with_video_analysis.csv"

    # results 폴더가 없으면 생성
    os.makedirs("results", exist_ok=True)

    try:
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 최종 병합 데이터가 저장되었습니다: {output_file}")
        print(f"   파일 크기: {len(final_df)} 행 x {len(final_df.columns)} 컬럼")
        return final_df
    except Exception as e:
        print(f"❌ 오류: 파일 저장 실패: {e}")
        return None

if __name__ == "__main__":
    merge_analysis_data()
