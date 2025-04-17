import streamlit as st
import numpy as np
import math
import plotly.graph_objects as go
from collections import Counter

# --- 수학 계산 함수 ---

def combinations(n, k):
    """ 조합(nCk) 계산 함수 """
    if k < 0 or k > n:
        return 0
    if k > n // 2:
        k = n - k
    res = 1
    for i in range(k):
        # Use integer division // to prevent potential float issues if intermediate results are large
        res = res * (n - i) // (i + 1)
    return res

def hypergeometric_prob(N_pop, K_success_pop, n_sample, k_success_sample):
    """
    초기하분포 확률 계산 P(X=k)
    N_pop: 모집단 크기 (여기서는 그룹 크기 G)
    K_success_pop: 모집단 내 성공 상태 수 (첫 샘플 크기 k)
    n_sample: 표본 크기 (다음 샘플 크기 k)
    k_success_sample: 표본 내 성공 상태 수 (겹치는 아이템 수)
    """
    comb_Kk = combinations(K_success_pop, k_success_sample)
    comb_NK_nk = combinations(N_pop - K_success_pop, n_sample - k_success_sample)
    comb_Nn = combinations(N_pop, n_sample) # G개 중 k개 뽑는 경우

    if comb_Nn == 0:
        return 0.0

    try:
         # Ensure calculation maintains precision and handles potential large numbers safely
         # Calculate probability term by term if necessary, or use log probabilities for stability
         # For now, direct calculation with checks
         if comb_Kk == 0 or comb_NK_nk == 0: # If either term is zero, probability is zero
              return 0.0
         prob = (comb_Kk * comb_NK_nk) / comb_Nn
         # Clamp result between 0 and 1 due to potential floating point inaccuracies
         return max(0.0, min(1.0, prob))
    except OverflowError:
        # Fallback or error indication if numbers are too large
        # st.error("계산 중 오버플로우 발생! 파라미터 값을 줄여보세요.") # Avoid calling st functions inside calculation logic
        print("OverflowError in hypergeometric_prob calculation") # Log error instead
        return float('nan')
    except ZeroDivisionError:
         return 0.0


def calculate_expected_prob(G_group_size, k_items_per_pull, PD_threshold):
    """
    이론적인 중복 확률 계산 P(X >= PD_threshold) - 단일 그룹 내 기준
    G_group_size: 그룹 크기 (새로운 모집단 크기)
    k_items_per_pull: 각 샘플에서 뽑는 아이템 수
    PD_threshold: 겹침 기준 갯수
    """
    N_pop = G_group_size
    K_success_pop = k_items_per_pull
    n_sample = k_items_per_pull

    prob_less_than_PD = 0.0
    for k_success in range(PD_threshold):
        prob_k = hypergeometric_prob(N_pop, K_success_pop, n_sample, k_success)
        if math.isnan(prob_k): # Handle potential calculation errors
             st.error("이론 확률 계산 중 오류 발생. 파라미터 값을 확인하거나 줄여보세요.")
             return float('nan')
        prob_less_than_PD += prob_k

    prob_ge_PD = 1.0 - prob_less_than_PD
    # Ensure probability is within valid range [0, 1]
    return max(0.0, min(1.0, prob_ge_PD))

# --- Streamlit 앱 구성 ---

st.set_page_config(layout="wide")

st.title("🔄 순차 그룹 콘텐츠 중복 확률 시뮬레이터")
st.write("""
사용자가 순서대로 콘텐츠를 소비하는 시나리오를 가정합니다.
전체 콘텐츠는 여러 개의 **순차적 그룹**으로 나뉘며, 사용자는 **각 그룹 내에서** 여러 번 콘텐츠 샘플을 뽑아봅니다.
이 앱은 **하나의 그룹 내**에서 샘플 간 중복이 발생할 이론적 확률과 시뮬레이션 빈도를 보여주고,
사용자가 **전체 그룹을 모두 거쳤을 때**의 평균적인 중복 발생 빈도를 시뮬레이션합니다.
""")

# --- 입력 파라미터 (사이드바) ---
st.sidebar.header("⚙️ 파라미터 설정")

# N: 전체 아이템 수
전체_아이템_수 = st.sidebar.number_input(
    "1. 전체 콘텐츠(아이템) 개수 (N)",
    min_value=10,
    value=1000,
    step=100,
    help="사용자가 소비할 수 있는 총 콘텐츠의 수입니다."
)

# G: 그룹 크기
그룹_크기 = st.sidebar.number_input(
    "2. 순차 그룹의 크기 (G)",
    min_value=10,
    value=50,
    step=10,
    help="전체 콘텐츠를 순서대로 몇 개씩 묶어 그룹을 만들지 결정합니다. 확률 계산 및 시뮬레이션의 기본 단위(모집단)가 됩니다."
)

# k: 그룹 내에서 한 번에 뽑는 아이템 수
뽑는_아이템_수 = st.sidebar.number_input(
    "3. 그룹 내에서 한 번에 뽑는 콘텐츠 개수 (k)",
    min_value=1,
    # Ensure k cannot be larger than G
    max_value=그룹_크기 if 그룹_크기 > 0 else 1,
    value=min(10, 그룹_크기) if 그룹_크기 > 0 else 1, # Default k to 10 or G if G < 10
    step=1,
    help="하나의 그룹 내에서 콘텐츠 샘플을 만들 때 몇 개를 선택할지 결정합니다."
)

# NP: 그룹 내 샘플링 반복 횟수
그룹내_샘플링_횟수 = st.sidebar.number_input(
    "4. 그룹 내 샘플링 반복 횟수 (NP: Number of Pulling per Group)",
    min_value=2, # Need at least 2 pulls to compare
    value=10,
    step=1,
    help="하나의 그룹 내에서 k개의 콘텐츠를 뽑는 행위를 몇 번 반복할지 결정합니다."
)

# PD: 겹침 기준 개수
겹침_기준_갯수 = st.sidebar.number_input(
    "5. 겹침 기준 개수 (PD: Number of Duplicated)",
    min_value=1,
     # Ensure PD cannot be larger than k
    max_value=뽑는_아이템_수 if 뽑는_아이템_수 > 0 else 1,
    value=min(3, 뽑는_아이템_수) if 뽑는_아이템_수 > 0 else 1, # Default PD to 3 or k if k < 3
    step=1,
    help=f"샘플들 간에 최소 몇 개 이상 겹쳐야 '중복 발생'으로 간주할지 설정합니다. (현재 {뽑는_아이템_수}개 뽑기 기준)"
)

# 시뮬레이션 실행 버튼
run_button = st.sidebar.button("🚀 시뮬레이션 실행!")

# --- 결과 저장용 변수 ---
# 세션 상태를 사용하여 시뮬레이션 결과를 저장 (버튼 재클릭 시 유지되도록)
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

# --- 결과 출력 (메인 영역) ---

if run_button:
    # 입력값 유효성 검사
    valid_input = True
    if 그룹_크기 <= 0 or 전체_아이템_수 <= 0 or 뽑는_아이템_수 <= 0 or 그룹내_샘플링_횟수 <= 1 or 겹침_기준_갯수 <= 0:
         st.error("모든 파라미터 값은 0보다 커야 하며, 그룹 내 샘플링 횟수(NP)는 2 이상이어야 합니다.")
         valid_input = False
    elif 그룹_크기 > 전체_아이템_수:
        st.error("그룹 크기(G)는 전체 아이템 수(N)보다 클 수 없습니다.")
        valid_input = False
    elif 뽑는_아이템_수 > 그룹_크기:
        st.error("한 번에 뽑는 개수(k)는 그룹 크기(G)보다 클 수 없습니다.")
        valid_input = False
    elif 겹침_기준_갯수 > 뽑는_아이템_수:
         st.error("겹침 기준 개수(PD)는 한 번에 뽑는 개수(k)보다 클 수 없습니다.")
         valid_input = False

    if valid_input:
        st.header("📊 결과 분석")
        results_data = {} # 결과를 저장할 딕셔너리

        # --- 1. 이론적 확률 계산 (단일 그룹 내 기준) ---
        st.subheader("📈 이론적 중복 확률 (EPD - 단일 그룹 내)")
        with st.spinner(f'그룹 크기 {그룹_크기} 안에서 {뽑는_아이템_수}개를 뽑는 샘플 2개 비교 시, {겹침_기준_갯수}개 이상 겹칠 이론적 확률 계산 중...'):
            기대_중복_확률 = calculate_expected_prob(그룹_크기, 뽑는_아이템_수, 겹침_기준_갯수)
            results_data['epd'] = 기대_중복_확률

        if not math.isnan(기대_중복_확률):
             st.metric(
                 label=f"최소 {겹침_기준_갯수}개 이상 겹칠 확률 (단일 그룹 내 예측)",
                 value=f"{기대_중복_확률:.2%}"
             )
             st.info(f"""
             이 값은 초기하분포를 기반으로 계산되었습니다.
             크기가 {그룹_크기}인 **하나의 그룹 안에서** {뽑는_아이템_수}개짜리 샘플을 두 개 만들었을 때,
             두 샘플 간에 {겹침_기준_갯수}개 이상의 아이템이 겹칠 수학적인 확률입니다.
             """)
        else:
             # 오류 메시지는 calculate_expected_prob 내부에서 출력됨
             pass


        # --- 2. 단일 그룹 시뮬레이션 (RPD) ---
        st.subheader("🔬 단일 그룹 시뮬레이션 결과 (RPD)")

        아이템_풀_단일그룹 = list(range(그룹_크기)) # 0 ~ G-1
        겹침_횟수_목록_단일 = []
        중복_발생_횟수_단일 = 0
        # 단일 그룹 내에서 생성된 샘플 목록 저장
        단일그룹_생성_샘플 = []
        비교_횟수_단일 = 그룹내_샘플링_횟수 - 1

        status_text_single = st.empty()
        if 그룹내_샘플링_횟수 > 1:
            try:
                # 첫 샘플 생성 및 저장
                첫_샘플_단일_set = set(np.random.choice(아이템_풀_단일그룹, 뽑는_아이템_수, replace=False))
                단일그룹_생성_샘플.append(sorted(list(첫_샘플_단일_set)))

                # 나머지 샘플 생성, 비교 및 저장
                for _ in range(비교_횟수_단일):
                    다음_샘플_단일_set = set(np.random.choice(아이템_풀_단일그룹, 뽑는_아이템_수, replace=False))
                    단일그룹_생성_샘플.append(sorted(list(다음_샘플_단일_set))) # 저장

                    겹침_개수 = len(첫_샘플_단일_set.intersection(다음_샘플_단일_set))
                    겹침_횟수_목록_단일.append(겹침_개수)
                    if 겹침_개수 >= 겹침_기준_갯수:
                        중복_발생_횟수_단일 += 1
                status_text_single.success("단일 그룹 시뮬레이션 완료!")
                results_data['rpd_single'] = 중복_발생_횟수_단일 / 비교_횟수_단일 if 비교_횟수_단일 > 0 else 0.0
                results_data['samples_single'] = 단일그룹_생성_샘플
                results_data['overlaps_single'] = 겹침_횟수_목록_단일

            except ValueError as e:
                 st.error(f"단일 그룹 시뮬레이션 중 오류 발생: {e}. 입력 값 확인 (예: k <= G).")
                 results_data['rpd_single'] = float('nan') # 오류 표시
        else:
             status_text_single.warning("그룹 내 샘플링 횟수(NP)가 2 이상이어야 비교 가능합니다.")
             results_data['rpd_single'] = float('nan') # 비교 불가 표시

        # 단일 그룹 결과 표시
        if not math.isnan(results_data.get('rpd_single', float('nan'))):
            st.metric(
                label=f"최소 {겹침_기준_갯수}개 이상 겹친 비율 (단일 그룹 시뮬레이션)",
                value=f"{results_data['rpd_single']:.2%}"
            )
            st.info(f"""
            **하나의 그룹(크기 {그룹_크기}) 안에서** {뽑는_아이템_수}개짜리 샘플을 {그룹내_샘플링_횟수}번 생성했습니다.
            첫 번째 샘플과 나머지 {비교_횟수_단일}개의 샘플을 비교한 결과,
            {중복_발생_횟수_단일}번의 경우에서 {겹침_기준_갯수}개 이상의 아이템이 겹쳤습니다.
            """)

            # 단일 그룹 겹침 분포 시각화
            if results_data.get('overlaps_single'):
                hist_fig_single = go.Figure(data=[go.Histogram(
                    x=results_data['overlaps_single'], name='겹침 개수',
                    xbins=dict(start=-0.5, end=뽑는_아이템_수 + 0.5, size=1),
                    marker_color='#636EFA'
                )])
                hist_fig_single.update_layout(
                    title_text='단일 그룹 내 겹침 개수 분포 (첫 샘플 기준)',
                    xaxis_title_text='겹친 아이템 개수', yaxis_title_text='빈도 (횟수)',
                    bargap=0.1
                )
                hist_fig_single.update_xaxes(tickmode='linear', dtick=1)
                st.plotly_chart(hist_fig_single, use_container_width=True)

            # 단일 그룹 생성 샘플 보기 (Expander)
            with st.expander("🎲 단일 그룹 시뮬레이션 생성 샘플 보기"):
                 if results_data.get('samples_single'):
                      for idx, sample in enumerate(results_data['samples_single']):
                           # 실제 아이템 ID 대신 그룹 내 인덱스(0~G-1) 표시
                           st.text(f"  샘플 {idx+1} (그룹 내 인덱스): {sample}")
                 else:
                      st.text("생성된 샘플이 없습니다.")


        # --- 3. 통합 시뮬레이션 (Integration Simulation) ---
        st.subheader("🌐 통합 시뮬레이션 결과 (전체 그룹 여정)")
        st.write(f"사용자가 전체 {전체_아이템_수}개 콘텐츠를 {그룹_크기}개씩 묶인 그룹으로 순차적으로 모두 경험하는 시나리오입니다.")

        총_비교_횟수_통합 = 0
        총_중복_발생_횟수_통합 = 0
        총_그룹_수 = 전체_아이템_수 // 그룹_크기
        겹침_횟수_목록_통합 = []
        # 통합 시뮬레이션에서 생성된 모든 샘플 저장 (그룹별로)
        통합_시뮬레이션_샘플 = {} # { "Group 1": [[샘플1], [샘플2], ...], "Group 2": ... }

        if 전체_아이템_수 % 그룹_크기 != 0:
            st.warning(f"주의: 전체 아이템 수({전체_아이템_수})가 그룹 크기({그룹_크기})로 나누어 떨어지지 않습니다. 마지막 부분의 아이템은 시뮬레이션에서 제외됩니다. 총 {총_그룹_수}개의 완전한 그룹만 고려됩니다.")

        integration_progress_bar = st.progress(0)
        integration_status_text = st.empty()
        simulation_error = False # 오류 발생 플래그

        if 총_그룹_수 > 0 and 그룹내_샘플링_횟수 > 1:
            비교_횟수_그룹당 = 그룹내_샘플링_횟수 - 1

            for i in range(총_그룹_수):
                # 현재 그룹의 실제 아이템 ID 범위 정의
                아이템_풀_현재그룹 = list(range(i * 그룹_크기, (i + 1) * 그룹_크기))
                group_key = f"Group {i+1} (Items {i * 그룹_크기}~{(i + 1) * 그룹_크기 - 1})"
                통합_시뮬레이션_샘플[group_key] = []

                try:
                    # 첫 샘플 생성 (실제 ID 사용) 및 저장
                    첫_샘플_그룹_set = set(np.random.choice(아이템_풀_현재그룹, 뽑는_아이템_수, replace=False))
                    통합_시뮬레이션_샘플[group_key].append(sorted(list(첫_샘플_그룹_set)))

                    # 나머지 샘플 생성, 비교 및 저장
                    for _ in range(비교_횟수_그룹당):
                        다음_샘플_그룹_set = set(np.random.choice(아이템_풀_현재그룹, 뽑는_아이템_수, replace=False))
                        통합_시뮬레이션_샘플[group_key].append(sorted(list(다음_샘플_그룹_set))) # 저장

                        겹침_개수 = len(첫_샘플_그룹_set.intersection(다음_샘플_그룹_set))
                        겹침_횟수_목록_통합.append(겹침_개수)
                        총_비교_횟수_통합 += 1
                        if 겹침_개수 >= 겹침_기준_갯수:
                            총_중복_발생_횟수_통합 += 1

                except ValueError as e:
                    st.error(f"통합 시뮬레이션 그룹 {i+1} 처리 중 오류: {e}. 입력 값 확인.")
                    simulation_error = True
                    break # 오류 발생 시 통합 시뮬레이션 중단

                # 진행 상황 업데이트
                progress = (i + 1) / 총_그룹_수
                integration_progress_bar.progress(progress)
                integration_status_text.text(f"통합 시뮬레이션: 그룹 {i+1}/{총_그룹_수} 처리 중...")

            integration_progress_bar.empty()
            if not simulation_error:
                 integration_status_text.success(f"통합 시뮬레이션 완료! (총 {총_그룹_수}개 그룹 처리)")
                 results_data['rpd_integration'] = 총_중복_발생_횟수_통합 / 총_비교_횟수_통합 if 총_비교_횟수_통합 > 0 else 0.0
                 results_data['total_comparisons_integration'] = 총_비교_횟수_통합
                 results_data['total_overlaps_integration'] = 총_중복_발생_횟수_통합
                 results_data['overlaps_integration'] = 겹침_횟수_목록_통합
                 results_data['samples_integration'] = 통합_시뮬레이션_샘플 # 저장된 샘플 결과

        else:
            integration_status_text.warning("그룹이 없거나 그룹 내 샘플링 횟수가 2 미만이라 통합 시뮬레이션을 수행할 수 없습니다.")
            results_data['rpd_integration'] = float('nan')


        # 통합 시뮬레이션 결과 표시
        if not simulation_error and not math.isnan(results_data.get('rpd_integration', float('nan'))):
            st.metric(
                label=f"평균 중복 발생 비율 (전체 {총_그룹_수}개 그룹 여정)",
                value=f"{results_data['rpd_integration']:.2%}"
            )
            st.info(f"""
            사용자가 총 {총_그룹_수}개의 순차 그룹을 모두 거치며, 각 그룹 내에서 {그룹내_샘플링_횟수-1}번의 샘플 비교를 수행했습니다.
            총 {results_data['total_comparisons_integration']}번의 비교 중 {results_data['total_overlaps_integration']}번의 경우에서 {겹침_기준_갯수}개 이상의 아이템이 겹쳤습니다.
            이는 전체 여정 동안 평균적으로 겹침이 발생한 비율을 나타냅니다.
            """)

            # 전체 여정 겹침 분포 시각화
            if results_data.get('overlaps_integration'):
                hist_fig_integ = go.Figure(data=[go.Histogram(
                    x=results_data['overlaps_integration'], name='겹침 개수',
                    xbins=dict(start=-0.5, end=뽑는_아이템_수 + 0.5, size=1),
                    marker_color='#FF7F0E' # 다른 색상 사용
                )])
                hist_fig_integ.update_layout(
                    title_text=f'전체 여정({총_그룹_수}개 그룹) 중 발생한 겹침 개수 분포',
                    xaxis_title_text='겹친 아이템 개수', yaxis_title_text='빈도 (횟수)',
                    bargap=0.1
                )
                hist_fig_integ.update_xaxes(tickmode='linear', dtick=1)
                st.plotly_chart(hist_fig_integ, use_container_width=True)

                # 통계 요약 추가
                평균_겹침_통합 = np.mean(results_data['overlaps_integration']) if results_data['overlaps_integration'] else 0
                st.write(f"**전체 여정 평균 겹침 개수:** {평균_겹침_통합:.2f} 개")
                # 이론적 기대값과 비교
                if 그룹_크기 > 0 : # Avoid division by zero
                    이론적_기대값_단일 = 뽑는_아이템_수 * 뽑는_아이템_수 / 그룹_크기
                    st.write(f"**단일 그룹 내 이론적 평균 겹침 개수 (기댓값):** {이론적_기대값_단일:.2f} 개")

            # 통합 시뮬레이션 생성 샘플 보기 (Expander)
            with st.expander("📜 통합 시뮬레이션 생성 샘플 보기 (그룹별)"):
                 if results_data.get('samples_integration'):
                      for group_name, samples_in_group in results_data['samples_integration'].items():
                           st.markdown(f"**{group_name}:**")
                           # 그룹별 샘플 표시 (너무 많으면 일부만)
                           display_limit_samples = min(len(samples_in_group), 5) # 그룹당 최대 5개 샘플 표시
                           for idx, sample in enumerate(samples_in_group[:display_limit_samples]):
                                st.text(f"  샘플 {idx+1}: {sample}") # 실제 아이템 ID 포함
                           if len(samples_in_group) > display_limit_samples:
                                st.text(f"  ... (그룹 내 총 {len(samples_in_group)}개 샘플 중 {display_limit_samples}개만 표시)")
                           # st.markdown("---") # 그룹 간 구분선 (선택 사항)
                 else:
                      st.text("생성된 샘플 데이터가 없습니다.")


        elif simulation_error:
             st.error("오류로 인해 통합 시뮬레이션 결과를 표시할 수 없습니다.")

        # --- 참고 정보 섹션 ---
        st.markdown("---")
        st.subheader("참고 정보")
        if 총_그룹_수 > 0:
            # 총 '샘플(묶음)' 수 계산 수정: (N/G) * NP
            총_생성_샘플_수 = 총_그룹_수 * 그룹내_샘플링_횟수
            st.write(f"사용자가 전체 여정 ({총_그룹_수}개 그룹)을 완료하며 각 그룹에서 {그룹내_샘플링_횟수}번 샘플링(각 {뽑는_아이템_수}개씩)할 경우, 생성되는 총 샘플(묶음) 수는 **{총_생성_샘플_수}개** 입니다.")
        else:
             st.write("그룹이 생성되지 않아 총 샘플 수를 계산할 수 없습니다.")

        if results_data.get('total_comparisons_integration') is not None:
             st.write(f"통합 시뮬레이션에서 수행된 총 비교 횟수는 **{results_data['total_comparisons_integration']}번** 입니다 (각 그룹 내 첫 샘플과 나머지 샘플 비교).")
        else:
             st.write("통합 시뮬레이션 비교가 수행되지 않았습니다.")

        # --- 세션 상태에 결과 저장 ---
        st.session_state.simulation_results = results_data

    # If valid_input is False, the errors would have been displayed above.
else:
    # 버튼을 누르지 않았을 때, 이전 결과가 있으면 표시 (선택 사항)
    if st.session_state.simulation_results:
        st.info("이전 시뮬레이션 결과입니다. 새 결과를 보려면 파라미터를 조정하고 버튼을 다시 누르세요.")
        # (Optional) Display previous results here if desired
        # This part requires more logic to redisplay formatted results from session state
    else:
        st.info("사이드바에서 파라미터를 설정하고 '시뮬레이션 실행!' 버튼을 눌러주세요.")


st.markdown("---")
st.caption("Streamlit App by Gemini")