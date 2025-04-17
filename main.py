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
    # G-k 개의 '실패' 아이템 중에서 k-겹침개수 만큼 뽑는 경우
    comb_NK_nk = combinations(N_pop - K_success_pop, n_sample - k_success_sample)
    comb_Nn = combinations(N_pop, n_sample) # G개 중 k개 뽑는 경우

    if comb_Nn == 0:
        return 0.0

    # 부동소수점 오차 방지를 위해 로그 변환 등 고려 가능하나, 여기서는 직접 계산
    # 매우 큰 수가 될 경우 Overflow 가능성 있음 (Streamlit 환경 고려)
    try:
        # 개별 조합 값이 너무 크면 Overflow 발생 가능
        # 확률 계산 시 분모/분자를 나눠서 계산하는 것이 더 안정적일 수 있음
        # prob = (comb_Kk / comb_Nn) * comb_NK_nk # 이 방식은 comb_NK_nk가 클 때 부정확 가능
        # 따라서 원래 공식 유지하되, 결과가 0~1 벗어나면 예외처리 필요
         prob = (comb_Kk * comb_NK_nk) / comb_Nn
         return max(0.0, min(1.0, prob)) # 확률은 0~1 사이
    except OverflowError:
        st.error("계산 중 오버플로우 발생! 파라미터 값을 줄여보세요.")
        return float('nan') # 계산 불가 시 NaN 반환
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
        if math.isnan(prob_k): # 계산 불가 시 중단
             return float('nan')
        prob_less_than_PD += prob_k

    prob_ge_PD = 1.0 - prob_less_than_PD
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
    max_value=그룹_크기, # 최대 그룹 크기까지 가능
    value=10,
    step=1,
    help="하나의 그룹 내에서 콘텐츠 샘플을 만들 때 몇 개를 선택할지 결정합니다."
)

# NP: 그룹 내 샘플링 반복 횟수
그룹내_샘플링_횟수 = st.sidebar.number_input(
    "4. 그룹 내 샘플링 반복 횟수 (NP: Number of Pulling per Group)",
    min_value=2,
    value=10,
    step=1,
    help="하나의 그룹 내에서 k개의 콘텐츠를 뽑는 행위를 몇 번 반복할지 결정합니다."
)

# PD: 겹침 기준 개수
겹침_기준_갯수 = st.sidebar.number_input(
    "5. 겹침 기준 개수 (PD: Number of Duplicated)",
    min_value=1,
    max_value=뽑는_아이템_수, # 최대 뽑는 아이템 수까지 가능
    value=3,
    step=1,
    help=f"샘플들 간에 최소 몇 개 이상 겹쳐야 '중복 발생'으로 간주할지 설정합니다. (현재 {뽑는_아이템_수}개 뽑기 기준)"
)

# 시뮬레이션 실행 버튼
run_button = st.sidebar.button("🚀 시뮬레이션 실행!")

# --- 결과 출력 (메인 영역) ---

if run_button:
    # 입력값 유효성 검사 (간단하게)
    if 그룹_크기 > 전체_아이템_수:
        st.error("그룹 크기(G)는 전체 아이템 수(N)보다 클 수 없습니다.")
    elif 뽑는_아이템_수 > 그룹_크기:
         st.error("한 번에 뽑는 개수(k)는 그룹 크기(G)보다 클 수 없습니다.")
    else:
        st.header("📊 결과 분석")

        # --- 1. 이론적 확률 계산 (단일 그룹 내 기준) ---
        st.subheader("📈 이론적 중복 확률 (EPD - 단일 그룹 내)")

        # 계산 시작
        with st.spinner(f'그룹 크기 {그룹_크기} 안에서 {뽑는_아이템_수}개를 뽑는 샘플 2개 비교 시, {겹침_기준_갯수}개 이상 겹칠 이론적 확률 계산 중...'):
            기대_중복_확률 = calculate_expected_prob(그룹_크기, 뽑는_아이템_수, 겹침_기준_갯수)

        if not math.isnan(기대_중복_확률):
             st.metric(
                 label=f"최소 {겹침_기준_갯수}개 이상 겹칠 확률 (단일 그룹 내 예측)",
                 value=f"{기대_중복_확률:.2%}"
             )
             st.info(f"""
             이 값은 초기하분포를 기반으로 계산되었습니다.
             크기가 {그룹_크기}인 **하나의 그룹 안에서** {뽑는_아이템_수}개짜리 샘플을 두 개 만들었을 때,
             두 샘플 간에 {겹침_기준_갯수}개 이상의 아이템이 겹칠 수학적인 확률입니다.
             어떤 순차 그룹이든 크기가 같다면 이 확률은 동일합니다.
             """)
        else:
             st.error("이론적 확률 계산 중 오류가 발생했습니다. 입력 값을 확인해주세요.")


        # --- 2. 단일 그룹 시뮬레이션 (RPD) ---
        st.subheader("🔬 단일 그룹 시뮬레이션 결과 (RPD)")

        아이템_풀_단일그룹 = list(range(그룹_크기)) # 0 ~ G-1
        겹침_횟수_목록_단일 = []
        중복_발생_횟수_단일 = 0
        비교_횟수_단일 = 그룹내_샘플링_횟수 - 1

        status_text_single = st.empty()
        if 그룹내_샘플링_횟수 > 1:
            try:
                첫_샘플_단일 = set(np.random.choice(아이템_풀_단일그룹, 뽑는_아이템_수, replace=False))
                for _ in range(비교_횟수_단일):
                    다음_샘플_단일 = set(np.random.choice(아이템_풀_단일그룹, 뽑는_아이템_수, replace=False))
                    겹침_개수 = len(첫_샘플_단일.intersection(다음_샘플_단일))
                    겹침_횟수_목록_단일.append(겹침_개수)
                    if 겹침_개수 >= 겹침_기준_갯수:
                        중복_발생_횟수_단일 += 1
                status_text_single.success("단일 그룹 시뮬레이션 완료!")
            except ValueError as e:
                 st.error(f"단일 그룹 시뮬레이션 중 오류 발생: {e}. 입력 값을 확인해주세요 (예: k <= G).")
                 중복_발생_횟수_단일 = -1 # 오류 플래그
        else:
             status_text_single.warning("그룹 내 샘플링 횟수(NP)가 2 이상이어야 비교 가능합니다.")
             중복_발생_횟수_단일 = -1 # 비교 불가 플래그

        if 중복_발생_횟수_단일 >= 0 and 비교_횟수_단일 > 0:
            실제_중복_확률_단일 = 중복_발생_횟수_단일 / 비교_횟수_단일
            st.metric(
                label=f"최소 {겹침_기준_갯수}개 이상 겹친 비율 (단일 그룹 시뮬레이션)",
                value=f"{실제_중복_확률_단일:.2%}"
            )
            st.info(f"""
            **하나의 그룹(크기 {그룹_크기}) 안에서** {뽑는_아이템_수}개짜리 샘플을 {그룹내_샘플링_횟수}번 생성했습니다.
            첫 번째 샘플과 나머지 {비교_횟수_단일}개의 샘플을 비교한 결과,
            {중복_발생_횟수_단일}번의 경우에서 {겹침_기준_갯수}개 이상의 아이템이 겹쳤습니다.
            """)

            # 단일 그룹 겹침 분포 시각화
            if 겹침_횟수_목록_단일:
                hist_fig_single = go.Figure(data=[go.Histogram(
                    x=겹침_횟수_목록_단일, name='겹침 개수',
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


        # --- 3. 통합 시뮬레이션 (Integration Simulation) ---
        st.subheader("🌐 통합 시뮬레이션 결과 (전체 그룹 여정)")
        st.write(f"사용자가 전체 {전체_아이템_수}개 콘텐츠를 {그룹_크기}개씩 묶인 그룹으로 순차적으로 모두 경험하는 시나리오입니다.")

        총_비교_횟수_통합 = 0
        총_중복_발생_횟수_통합 = 0
        총_그룹_수 = 전체_아이템_수 // 그룹_크기
        겹침_횟수_목록_통합 = []

        if 전체_아이템_수 % 그룹_크기 != 0:
            st.warning(f"주의: 전체 아이템 수({전체_아이템_수})가 그룹 크기({그룹_크기})로 나누어 떨어지지 않습니다. 마지막 부분의 아이템은 시뮬레이션에서 제외됩니다. 총 {총_그룹_수}개의 완전한 그룹만 고려됩니다.")

        integration_progress_bar = st.progress(0)
        integration_status_text = st.empty()

        if 총_그룹_수 > 0 and 그룹내_샘플링_횟수 > 1:
            비교_횟수_그룹당 = 그룹내_샘플링_횟수 - 1
            아이템_풀_그룹 = list(range(그룹_크기)) # 각 그룹 내 아이템 인덱스 (0 ~ G-1)

            for i in range(총_그룹_수):
                # 현재 그룹 내 시뮬레이션
                try:
                    첫_샘플_그룹 = set(np.random.choice(아이템_풀_그룹, 뽑는_아이템_수, replace=False))
                    for _ in range(비교_횟수_그룹당):
                        다음_샘플_그룹 = set(np.random.choice(아이템_풀_그룹, 뽑는_아이템_수, replace=False))
                        겹침_개수 = len(첫_샘플_그룹.intersection(다음_샘플_그룹))
                        겹침_횟수_목록_통합.append(겹침_개수) # 전체 여정의 겹침 분포 위해 저장
                        총_비교_횟수_통합 += 1
                        if 겹침_개수 >= 겹침_기준_갯수:
                            총_중복_발생_횟수_통합 += 1
                except ValueError as e:
                    st.error(f"통합 시뮬레이션 그룹 {i+1} 처리 중 오류: {e}. 입력 값 확인.")
                    # 오류 발생 시 해당 그룹은 건너뛰거나 중단 처리 가능
                    총_중복_발생_횟수_통합 = -1 # 오류 플래그
                    break # 통합 시뮬레이션 중단

                # 진행 상황 업데이트
                progress = (i + 1) / 총_그룹_수
                integration_progress_bar.progress(progress)
                integration_status_text.text(f"통합 시뮬레이션: 그룹 {i+1}/{총_그룹_수} 처리 중...")

            integration_progress_bar.empty()
            if 총_중복_발생_횟수_통합 >= 0 :
                 integration_status_text.success(f"통합 시뮬레이션 완료! (총 {총_그룹_수}개 그룹 처리)")

        else:
            integration_status_text.warning("그룹이 없거나 그룹 내 샘플링 횟수가 2 미만이라 통합 시뮬레이션을 수행할 수 없습니다.")
            총_중복_발생_횟수_통합 = -1 # 수행 불가 플래그


        if 총_중복_발생_횟수_통합 >= 0 and 총_비교_횟수_통합 > 0:
            평균_중복_비율_통합 = 총_중복_발생_횟수_통합 / 총_비교_횟수_통합
            st.metric(
                label=f"평균 중복 발생 비율 (전체 {총_그룹_수}개 그룹 여정)",
                value=f"{평균_중복_비율_통합:.2%}"
            )
            st.info(f"""
            사용자가 총 {총_그룹_수}개의 순차 그룹을 모두 거치며, 각 그룹 내에서 {그룹내_샘플링_횟수-1}번의 샘플 비교를 수행했습니다.
            총 {총_비교_횟수_통합}번의 비교 중 {총_중복_발생_횟수_통합}번의 경우에서 {겹침_기준_갯수}개 이상의 아이템이 겹쳤습니다.
            이는 전체 여정 동안 평균적으로 겹침이 발생한 비율을 나타냅니다.
            """)

            # 전체 여정 겹침 분포 시각화
            if 겹침_횟수_목록_통합:
                hist_fig_integ = go.Figure(data=[go.Histogram(
                    x=겹침_횟수_목록_통합, name='겹침 개수',
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
                평균_겹침_통합 = np.mean(겹침_횟수_목록_통합) if 겹침_횟수_목록_통합 else 0
                st.write(f"**전체 여정 평균 겹침 개수:** {평균_겹침_통합:.2f} 개")
                # 이론적 기대값과 비교
                이론적_기대값_단일 = 뽑는_아이템_수 * 뽑는_아이템_수 / 그룹_크기
                st.write(f"**단일 그룹 내 이론적 평균 겹침 개수 (기댓값):** {이론적_기대값_단일:.2f} 개")
        elif 총_중복_발생_횟수_통합 == -1:
             st.error("오류로 인해 통합 시뮬레이션 결과를 표시할 수 없습니다.")

        # (1000/G) * NP 관련 부가 정보
        st.markdown("---")
        st.subheader("참고 정보")
        총_샘플수_추정 = 총_그룹_수 * 그룹내_샘플링_횟수 * 뽑는_아이템_수
        st.write(f"사용자가 전체 여정 ({총_그룹_수}개 그룹)을 완료하며 각 그룹에서 {그룹내_샘플링_횟수}번 샘플링할 경우, 이론적으로 생성되는 총 샘플 수는 **{총_샘플수_추정}개** 입니다.")
        st.write(f"통합 시뮬레이션에서 수행된 총 비교 횟수는 **{총_비교_횟수_통합}번** 입니다 (각 그룹 내 첫 샘플과 나머지 샘플 비교).")


else:
    st.info("사이드바에서 파라미터를 설정하고 '시뮬레이션 실행!' 버튼을 눌러주세요.")

st.markdown("---")
st.caption("Streamlit App by Gemini")