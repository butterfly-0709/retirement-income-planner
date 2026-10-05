import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm

# ===================== 基本参数 =====================
LTCI_PRICE = 81153.88
W0 = 407328.0
omega_L = LTCI_PRICE / W0  # LTCI allocation is fixed-price / W0
omega_G = 0.2
investable_wealth_initial = (1 - omega_L - omega_G) * W0
GMDB_initial_allocation = omega_G * W0
TARGET_VOLATILITY = 0.2

params = {
    "regime1": {"kappa": 0.08, "sigma": 0.20, "r": 0.04},
    "regime2": {"kappa": 0.04, "sigma": 0.40, "r": 0.02},
    "a12": 0.40, "a21": 0.30, "S0": 100.0, "B0": 100.0, "initial_state": 1
}

# --- 预先计算好的“健康状态”年转移矩阵（3x3，死亡为吸收态） ---
# ... (pre_computed_matrices_data 保持不变，此处省略，因为它很长)
pre_computed_matrices_data = [
    [[0.9355252, 0.05175833, 0.00900526, 0.003015795, 0.0006954245], [0.0, 0.96751240, 0.0, 0.022004735, 0.0104828671], [0.3670531, 0.02000043, 0.54877953, 0.025355152, 0.0388117977], [0.0, 0.34331576, 0.0, 0.599741715, 0.0569425301], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.9326658, 0.05371684, 0.009579418, 0.003261739, 0.0007761683], [0.0, 0.96562790, 0.0, 0.023186876, 0.0111852246], [0.3603820, 0.02014252, 0.551570916, 0.025885653, 0.0420189468], [0.0, 0.33970562, 0.0, 0.600033138, 0.0602612468], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.9296760, 0.05574244, 0.01018695, 0.003527166, 0.0008674131], [0.0, 0.96363613, 0.0, 0.024427829, 0.0119360413], [0.3537199, 0.02029247, 0.55408882, 0.026418046, 0.0454807509], [0.0, 0.33606882, 0.0, 0.600165650, 0.0637655312], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.9265498, 0.05783671, 0.01082942, 0.003813536, 0.0009705507], [0.0, 0.96153122, 0.0, 0.025729990, 0.0127387895], [0.3470657, 0.02044951, 0.55631720, 0.026951572, 0.0492160121], [0.0, 0.33240370, 0.0, 0.600131426, 0.0674648784], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.9232808, 0.06000121, 0.01150839, 0.004122402, 0.001087157], [0.0, 0.95930701, 0.0, 0.027095792, 0.013597200], [0.3404180, 0.02061288, 0.55823904, 0.027485390, 0.053244705], [0.0, 0.32870852, 0.0, 0.599922311, 0.071369165], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.9198627, 0.06223742, 0.01222546, 0.004455419, 0.001219014], [0.0, 0.95695702, 0.0, 0.028527696, 0.014515282], [0.3337753, 0.02078180, 0.55983634, 0.028018575, 0.057588011], [0.0, 0.32498154, 0.0, 0.599529813, 0.075488652], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.9162885, 0.06454679, 0.01298219, 0.004814347, 0.00136814], [0.0, 0.95447448, 0.0, 0.030028176, 0.01549734], [0.3271360, 0.02095545, 0.56109006, 0.028550112, 0.06226835], [0.0, 0.32122091, 0.0, 0.598945103, 0.07983398], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.9125513, 0.06693068, 0.01378017, 0.005201055, 0.001536816], [0.0, 0.95185228, 0.0, 0.031599711, 0.016548013], [0.3204986, 0.02113300, 0.56198009, 0.029078886, 0.067309416], [0.0, 0.31742480, 0.0, 0.598159016, 0.084416188], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.9086436, 0.06939035, 0.01462092, 0.005617524, 0.001727619], [0.0, 0.94908296, 0.0, 0.033244772, 0.017672266], [0.3138613, 0.02131358, 0.56248530, 0.029603678, 0.072736161], [0.0, 0.31359128, 0.0, 0.597162051, 0.089246670], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.9045578, 0.07192697, 0.01550595, 0.006065849, 0.001943462], [0.0, 0.94615875, 0.0, 0.034965806, 0.018875445], [0.3072223, 0.02149629, 0.56258345, 0.030123157, 0.078574830], [0.0, 0.30971842, 0.0, 0.595944375, 0.094337204], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.9002859, 0.07454156, 0.01643668, 0.006548247, 0.002187634], [0.0, 0.94307149, 0.0, 0.036765218, 0.020163296], [0.3005797, 0.02168021, 0.56225126, 0.030635875, 0.084852930], [0.0, 0.30580425, 0.0, 0.594495831, 0.099699922], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.8958196, 0.07723504, 0.01741444, 0.007067052, 0.002463843], [0.0, 0.93981266, 0.0, 0.038645351, 0.021541986], [0.2939318, 0.02186435, 0.56146442, 0.031140258, 0.091599208], [0.0, 0.30184675, 0.0, 0.592805949, 0.105347302], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.8911504, 0.08000810, 0.01844047, 0.007624723, 0.002776274], [0.0, 0.93637339, 0.0, 0.040608465, 0.023018145], [0.2872765, 0.02204769, 0.56019763, 0.031634600, 0.098843597], [0.0, 0.29784390, 0.0, 0.590863961, 0.111292143], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.8862694, 0.08286130, 0.01951582, 0.008223839, 0.003129638], [0.0, 0.93274440, 0.0, 0.042656713, 0.024598890], [0.2806120, 0.02222917, 0.55842464, 0.032117060, 0.106617145], [0.0, 0.29379364, 0.0, 0.588658814, 0.117547546], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.8811673, 0.08579494, 0.02064139, 0.008867101, 0.00352924], [0.0, 0.92891603, 0.0, 0.044792108, 0.02629186], [0.2739364, 0.02240767, 0.55611841, 0.032585649, 0.11495191], [0.0, 0.28969392, 0.0, 0.586179199, 0.12412688], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.8758347, 0.0888091, 0.02181786, 0.009557327, 0.003981044], [0.0, 0.9248782, 0.0, 0.047016497, 0.028105263], [0.2672478, 0.0225820, 0.55325114, 0.033038232, 0.123880830], [0.0, 0.2855427, 0.0, 0.583413575, 0.131043747], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.8702616, 0.09190358, 0.02304563, 0.01029745, 0.004491752], [0.0, 0.92062058, 0.0, 0.04933152, 0.030047894], [0.2605445, 0.02275095, 0.54979448, 0.03347252, 0.133437541], [0.0, 0.28133786, 0.0, 0.58035020, 0.138311942], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.8644379, 0.09507785, 0.02432481, 0.01109052, 0.005068883], [0.0, 0.91613221, 0.0, 0.05173859, 0.032129194], [0.2538248, 0.02291321, 0.54571972, 0.03388606, 0.143656170], [0.0, 0.27707742, 0.0, 0.57697719, 0.145945398], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.8583533, 0.09833107, 0.02565514, 0.01193966, 0.005720863], [0.0, 0.91140190, 0.0, 0.05423882, 0.034359280], [0.2470873, 0.02306743, 0.54099800, 0.03427626, 0.154571062], [0.0, 0.27275935, 0.0, 0.57328251, 0.153958134], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.8519968, 0.10166199, 0.02703594, 0.01284812, 0.006457121], [0.0, 0.90641800, 0.0, 0.05683301, 0.036748988], [0.2403304, 0.02321218, 0.53560058, 0.03464033, 0.166216469], [0.0, 0.26838170, 0.0, 0.56925412, 0.162364182], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.8453576, 0.10506896, 0.02846606, 0.01381920, 0.007288196], [0.0, 0.90116852, 0.0, 0.05952156, 0.039309923], [0.2335533, 0.02334598, 0.52949923, 0.03497538, 0.178626160], [0.0, 0.26394255, 0.0, 0.56487993, 0.171177513], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.8384243, 0.10854983, 0.02994379, 0.01485625, 0.008225843], [0.0, 0.89564106, 0.0, 0.06230445, 0.042054494], [0.2267549, 0.02346727, 0.52266658, 0.03527830, 0.191832983], [0.0, 0.25944008, 0.0, 0.56014797, 0.180411952], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.8311854, 0.11210198, 0.03146681, 0.01596268, 0.009283156], [0.0, 0.88982286, 0.0, 0.06518117, 0.044995969], [0.2199348, 0.02357443, 0.51507657, 0.03554590, 0.205868342], [0.0, 0.25487253, 0.0, 0.55504639, 0.190081077], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.8236291, 0.11572217, 0.03303211, 0.01714188, 0.01047469], [0.0, 0.88370084, 0.0, 0.06815065, 0.04814851], [0.2130928, 0.02366576, 0.50670504, 0.03577479, 0.22076160], [0.0, 0.25023829, 0.0, 0.54956360, 0.20019811], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.8157437, 0.11940660, 0.03463591, 0.01839724, 0.01181657], [0.0, 0.87726159, 0.0, 0.07121119, 0.05152722], [0.2062293, 0.02373953, 0.49753022, 0.03596152, 0.23653942], [0.0, 0.24553584, 0.0, 0.54368835, 0.21077580], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.8075169, 0.12315075, 0.03627359, 0.01973208, 0.01332667], [0.0, 0.87049139, 0.0, 0.07436041, 0.05514820], [0.1993451, 0.02379392, 0.48753353, 0.03610249, 0.25322495], [0.0, 0.24076386, 0.0, 0.53740985, 0.22182628], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.7989367, 0.12694939, 0.03793959, 0.02114962, 0.01502471], [0.0, 0.86337630, 0.0, 0.07759514, 0.05902856], [0.1924416, 0.02382708, 0.47670020, 0.03619406, 0.27083707], [0.0, 0.23592119, 0.0, 0.53071788, 0.23336093], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.7899909, 0.1307965, 0.03962735, 0.02265292, 0.01693237], [0.0, 0.8559021, 0.0, 0.08091138, 0.06318647], [0.1855208, 0.0238371, 0.46502017, 0.03623252, 0.28938938], [0.0, 0.2310069, 0.0, 0.52360294, 0.24539020], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.7806672, 0.13468519, 0.04132925, 0.02424485, 0.01907348], [0.0, 0.84805460, 0.0, 0.08430417, 0.06764123], [0.1785856, 0.02382205, 0.45248894, 0.03621416, 0.30888923], [0.0, 0.22602018, 0.0, 0.51605639, 0.25792343], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.7709537, 0.13860767, 0.04303652, 0.02592800, 0.02147408], [0.0, 0.83981924, 0.0, 0.08776755, 0.07241321], [0.1716396, 0.02377998, 0.43910850, 0.03613532, 0.32933664], [0.0, 0.22096070, 0.0, 0.50807060, 0.27096870], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.7608385, 0.14255516, 0.04473919, 0.02770460, 0.02416257], [0.0, 0.83118158, 0.0, 0.09129446, 0.07752396], [0.1646871, 0.02370893, 0.42488833, 0.03599240, 0.35072320], [0.0, 0.21582827, 0.0, 0.49963916, 0.28453257], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.7503098, 0.14651783, 0.04642607, 0.02957647, 0.02716978], [0.0, 0.82212721, 0.0, 0.09487662, 0.08299617], [0.1577338, 0.02360696, 0.40984641, 0.03578197, 0.37303088], [0.0, 0.21062310, 0.0, 0.49075700, 0.29861991], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.7393566, 0.15048473, 0.04808469, 0.03154493, 0.03052903], [0.0, 0.81264183, 0.0, 0.09850449, 0.08885368], [0.1507860, 0.02347217, 0.39401018, 0.03550080, 0.39623086], [0.0, 0.20534575, 0.0, 0.48142064, 0.31323361], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.7279680, 0.15444374, 0.04970138, 0.03361065, 0.03427619], [0.0, 0.80271139, 0.0, 0.10216717, 0.09512144], [0.1438513, 0.02330272, 0.37741758, 0.03514594, 0.42028243], [0.0, 0.19999720, 0.0, 0.47162838, 0.32837442], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.7161340, 0.15838150, 0.05126122, 0.03577362, 0.03844965], [0.0, 0.79232217, 0.0, 0.10585229, 0.10182554], [0.1369385, 0.02309687, 0.36011787, 0.03471484, 0.44513190], [0.0, 0.19457888, 0.0, 0.46138051, 0.34404062], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.7038453, 0.16228335, 0.05274816, 0.03803297, 0.04309026], [0.0, 0.78146094, 0.0, 0.10954599, 0.10899307], [0.1300575, 0.02285298, 0.34217248, 0.03420539, 0.47071166], [0.0, 0.18909271, 0.0, 0.45067950, 0.36022779], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.6910934, 0.16613331, 0.05414516, 0.04038689, 0.04824126], [0.0, 0.77011508, 0.0, 0.11323280, 0.11665212], [0.1232194, 0.02256961, 0.32365554, 0.03361603, 0.49693945], [0.0, 0.18354114, 0.0, 0.43953029, 0.37692857], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.6778712, 0.16991397, 0.0554343, 0.04283246, 0.05394806], [0.0, 0.75827274, 0.0, 0.11689563, 0.12483164], [0.1164365, 0.02224549, 0.3046543, 0.03294586, 0.52371783], [0.0, 0.17792719, 0.0, 0.42794046, 0.39413235], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.6641728, 0.17360654, 0.05659704, 0.04536556, 0.06025801], [0.0, 0.74592301, 0.0, 0.12051570, 0.13356129], [0.1097225, 0.02187962, 0.28526913, 0.03219473, 0.55093403], [0.0, 0.17225449, 0.0, 0.41592048, 0.41182502], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.649994, 0.17719077, 0.0576145, 0.04798069, 0.06722004], [0.0, 0.73305612, 0.0, 0.12407255, 0.14287133], [0.103092, 0.02147125, 0.2656132, 0.03136333, 0.57846019], [0.0, 0.16652732, 0.0, 0.40348393, 0.42998874], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.63533216, 0.1806450, 0.05846775, 0.05067087, 0.07488424], [0.0, 0.7196636, 0.0, 0.12754401, 0.15279235], [0.09656081, 0.0210200, 0.24581187, 0.03045329, 0.60615403], [0.0, 0.1607506, 0.0, 0.39064769, 0.44860166], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.62018685, 0.18394607, 0.05913829, 0.05342745, 0.08330135], [0.0, 0.70573869, 0.0, 0.13090625, 0.16335506], [0.09014561, 0.02052584, 0.22600107, 0.02946728, 0.63386021], [0.0, 0.15493015, 0.0, 0.37743214, 0.46763771], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.60455993, 0.18706951, 0.05960844, 0.05624001, 0.09252211], [0.0, 0.69127619, 0.0, 0.13413386, 0.17458995], [0.08386387, 0.01998917, 0.20632582, 0.02840902, 0.66141211], [0.0, 0.14907226, 0.0, 0.36386134, 0.48706640], [0.0, 0.0, 0.0, 0.0, 1.0]],
    [[0.58845585, 0.18998950, 0.05986189, 0.05909624, 0.1025965], [0.0, 0.67627311, 0.0, 0.13719988, 0.1865270], [0.07773361, 0.01941087, 0.18693764, 0.02728339, 0.6886345], [0.0, 0.14318419, 0.0, 0.34996316, 0.5068527], [0.0, 0.0, 0.0, 0.0, 1.0]]
]  
pre_computed_matrices = [np.array(m) for m in pre_computed_matrices_data]


#：0=健康, 1=生病, 2=健康残, 3=生病残, 4=死亡
HEALTH_COST_BASE_ANNUAL = {
    0: 0.0,   # 0
    1: 0.0,  # 1
    2: 50000.0,  # 2 
    3: 50000.0,  # 3
    4: 0.0  # 4
}
HEALTH_COST_INFLATION_Q = 0.019  # 1.9% 年通胀



# ===================== 函数定义 (保持不变) =====================

def annual_health_cost(state_idx: int, year_idx: int) -> float:
    """
    修改2：年度健康支出
    year_idx 从 0 开始计（第0年、1年、…）。slides 的公式是 h_k(t)=h_{k,1}(1+q)^{t-1}。
    因此这里用 (1+q)**year_idx。
    """
    base = HEALTH_COST_BASE_ANNUAL.get(state_idx, 0.0)
    return base * (1.0 + HEALTH_COST_INFLATION_Q) ** (year_idx)

def simulate_ctmc(T, dt, a12, a21, init_state):
    """ 连续时间两状态马氏链 → 日频路径 """
    N = int(T / dt)
    Q = np.array([[-a12,  a12],
                  [ a21, -a21]], dtype=float)
    P = expm(Q * dt)
    P = np.maximum(P, 0.0)  #储存
    P /= P.sum(axis=1, keepdims=True) #保证和为1，以避免影响输出

    states = np.zeros(N, dtype=int) #给输出的数储存的位置
    curr = init_state
    for i in range(N):
        states[i] = curr
        # Adjusting state for 0-indexing before accessing P
        curr_0_indexed = curr - 1 #这个要减去1因为索引是从0开始的
        curr = np.random.choice([0, 1], p=P[curr_0_indexed]) + 1 # +1 to return to 1-indexing for other parts
    return states

def simulate_asset_prices(params, T=33, dt=1/252):
    """ 体制切换下的股票/债券价格路径（日频） """
    N = int(T / dt) #N就是总的模拟天数
    S_price = np.zeros(N); B_price = np.zeros(N)  #给输出的数储存的位置
    S_price[0], B_price[0] = params["S0"], params["B0"]
    # regimes are 1-indexed here, as per simulate_ctmc output
    regimes = simulate_ctmc(T, dt, params["a12"], params["a21"], params["initial_state"]) #用到上面定义的simulate_ctmc公式得到状态
    for t in range(1, N):
        regime_idx = regimes[t - 1] # 1 or 2数组是 0 索引的，所以如果我们当前在时间步 t，取 t-1 作为当前时间步的体制状态
        p = params["regime1"] if regime_idx == 1 else params["regime2"] #每个状态对应不同的参数值引用，下面的p括号里面就是
        dW = np.random.normal(0, np.sqrt(dt)) #定义布朗运动
        S_price[t] = S_price[t - 1] * np.exp((p["kappa"] - 0.5 * p["sigma"]**2) * dt + p["sigma"] * dW) #公式
        B_price[t] = B_price[t - 1] * np.exp(p["r"] * dt) #公式
    return S_price, B_price, regimes #输出

def calculate_dynamic_allocations_monthly_true(S_price_path, TV, dt_day=1/252, days_per_month=21,
                                               lambda_ewma=0.8, initial_vol=0.20):
    """每个“月初日”用 S_t / S_{t-21} 的月度收益更新一次年化方差，年化步长用 dt_mon=1/12。整月内 α 固定为该月初。 """
    N = len(S_price_path)
    dt_mon = 1/12  # 月频步长（年化）
   # 先标出“月初日”索引（第0天、21、42、...）
    month_starts = list(range(0, N, days_per_month))
    if month_starts[-1] != N-1:
        month_starts.append(min(N-1, month_starts[-1] + days_per_month))
    # 结果（按日输出，月内用同一个α与σ）
    sigma2_t = np.zeros(N)
    sigma_hat_t = np.zeros(N)
    alpha_t = np.zeros(N)
    # 初始化
    sigma2 = initial_vol**2
    sigma_hat = initial_vol
    alpha = min(TV / sigma_hat, 1.0)
    # 将第一个月整段填上初始值
    first_month_end = month_starts[1] if len(month_starts) > 1 else N
    sigma2_t[:first_month_end] = sigma2
    sigma_hat_t[:first_month_end] = sigma_hat
    alpha_t[:first_month_end] = alpha
    # 从第二个“月初日”开始，按月更新
    for i in range(1, len(month_starts)):
        t0 = month_starts[i]                 # 本月月初日
        t_prev = t0 - days_per_month         # 上月月初日（保证 t_prev >= 0）
        # 月度对数收益（上月月初 → 本月月初）
        r_m = np.log(S_price_path[t0] / S_price_path[t_prev])
        # EWMA（年化方差）
        sigma2 = lambda_ewma * sigma2 + (1 - lambda_ewma) * (r_m**2) / dt_mon
        sigma_hat = np.sqrt(sigma2)
        # 目标波动率映射：α_m = min(TV/σ̂_m, 1)
        alpha = min(TV / sigma_hat, 1.0)
        # 将“本月整段”的日索引都填上同一个 α 与 σ
        t1 = month_starts[i+1] if i+1 < len(month_starts) else N
        sigma2_t[t0:t1] = sigma2
        sigma_hat_t[t0:t1] = sigma_hat
        alpha_t[t0:t1] = alpha
    return alpha_t, sigma_hat_t

def compute_discount_factors_from_regimes(regimes, params, dt=1/252):
    """ 按体制利率路径生成日贴现因子 D_t """
    # regimes are 1-indexed (1 or 2), convert to 0-indexed for params lookup
    r_t_series = np.array([params[f"regime{r}"]["r"] for r in regimes])
    return np.exp(-np.cumsum(r_t_series) * dt)

def compute_annuity_due_by_survival_discount(
        t_year,
        matrices,
        current_health_state,
        discount_factors,
        dt
):
    """
    计算给定当前健康状态下的 survival-contingent annuity-due factor。

    定义：
        a_due(t, i)
        = sum_{k >= 0}
          [D(t+k) / D(t)]
          * P(alive at t+k | X_t = i)

    健康状态：
        0, 1, 2, 3 = alive
        4 = dead

    注意：
        1. 当前健康状态只决定未来状态分布的起点。
        2. 未来只要处于 0,1,2,3 中任何状态，都视为 alive。
        3. k=0 包含当前时点，因此这是 annuity-due。
        4. 保持原函数接口和 float 返回值，不影响后续代码。
    """

    # ---------------------------------------------------------
    # 1. 基本检查
    # ---------------------------------------------------------
    if current_health_state == 4:
        return 0.0

    if current_health_state not in (0, 1, 2, 3):
        raise ValueError(
            f"Invalid health state: {current_health_state}"
        )

    if t_year < 0 or t_year >= len(matrices):
        return 0.0

    # 每年对应的日频步数
    N_per_year = int(round(1.0 / dt))

    # 当前年份在日频 discount factor 中的位置
    current_discount_idx = t_year * N_per_year

    if current_discount_idx >= len(discount_factors):
        return 0.0

    # 当前时点绝对贴现因子
    D_t = discount_factors[current_discount_idx]

    if D_t <= 0:
        raise ValueError("Discount factor must be positive.")

    # ---------------------------------------------------------
    # 2. 当前健康状态作为条件分布的起点
    #
    # 例如：
    # state 0 -> [1,0,0,0,0]
    # state 2 -> [0,0,1,0,0]
    # ---------------------------------------------------------
    p_state = np.zeros(5, dtype=float)
    p_state[current_health_state] = 1.0

    annuity_value = 0.0

    # ---------------------------------------------------------
    # 3. k = 0,1,2,...
    #
    # len(matrices)-t_year 个剩余 transition matrices
    # 可以产生 len(matrices)-t_year+1 个支付时点：
    #
    # k = 0                    当前支付
    # ...
    # k = len(matrices)-t_year 最后一次转移后的支付
    # ---------------------------------------------------------
    max_k = len(matrices) - t_year

    for k in range(max_k + 1):

        future_year = t_year + k

        # -----------------------------------------------------
        # k > 0 时，先从前一个年龄向当前 future_year 转移
        #
        # k=1: 使用 matrices[t_year]
        # k=2: 再使用 matrices[t_year+1]
        # ...
        # -----------------------------------------------------
        if k > 0:
            transition_year = future_year - 1

            if transition_year >= len(matrices):
                break

            p_state = (
                p_state
                @ matrices[transition_year]
            )

        # -----------------------------------------------------
        # 4. 生存概率
        #
        # 无论当前最初是什么健康状态，
        # 未来 0,1,2,3 全部属于 alive。
        # -----------------------------------------------------
        survival_probability = np.sum(p_state[:4])

        # 数值误差保护
        survival_probability = np.clip(
            survival_probability,
            0.0,
            1.0
        )

        # -----------------------------------------------------
        # 5. 找到 future_year 对应的日频贴现因子
        # -----------------------------------------------------
        future_discount_idx = future_year * N_per_year

        if future_discount_idx >= len(discount_factors):
            break

        D_future = discount_factors[future_discount_idx]

        # 从 future_year 相对于当前 t_year 进行贴现
        relative_discount = D_future / D_t

        # -----------------------------------------------------
        # 6. 加入 annuity-due PV
        # -----------------------------------------------------
        annuity_value += (
            relative_discount
            * survival_probability
        )

    return float(annuity_value)



def _month_start_discount(D_daily: np.ndarray, days_per_month: int = 21) -> np.ndarray:
    """
    将日度贴现因子 D_daily 下采样为“每月月初”的贴现序列。
    返回长度 ~ (T*12 + 1) 的序列（含 t=0 的 D_month[0]）。
    """
    N = len(D_daily)
    month_idxs = list(range(0, N, days_per_month))
    if month_idxs[-1] != N - 1:
        month_idxs.append(N - 1)
    return D_daily[np.array(month_idxs, dtype=int)]


def income_test(monthly_wealth, is_annuity_income=False, fixed_monthly_annuity_value=0.0):
    """
    根据个人金融资产计算养老金削减额（收入测试）。
    :param monthly_wealth: 当前月的金融资产总额。
    :param is_annuity_income: 标记是否为年金收入（C_fixed_monthly_list对应的策略）。
    :param fixed_monthly_annuity_value: 如果是年金收入，传入固定的每月年金值。
    :return: 经过收入测试后的每月养老金。
    """
    basic_pension = 2566.36  # 每月基础养老金

    if is_annuity_income:
        # 年金收入直接视为月收入，不进行推定收益率计算
        monthly_income = fixed_monthly_annuity_value
    else:
        # 一般资产，按推定收益率计算预期年收入，再折算每月
        threshold_tier_1 = 64200 / 12 # 对应每月资产，6.42万澳元以下
        rate_tier_1 = 0.0075 / 12   # 0.75% 年化，折算每月
        rate_tier_2 = 0.0275 / 12   # 2.75% 年化，折算每月

        if monthly_wealth <= threshold_tier_1:
            annual_deemed_income = monthly_wealth * rate_tier_1 * 12
        else:
            annual_deemed_income = (threshold_tier_1 * rate_tier_1 +
                                    (monthly_wealth - threshold_tier_1) * rate_tier_2) * 12
        monthly_income = annual_deemed_income / 12 # 再次折算月收入

    # 养老金起减点
    income_free_area = 475 # 每月475澳元
    pension_reduction_rate = 0.5 # 每超出1澳元减少0.5澳元

    if monthly_income <= income_free_area:
        return basic_pension
    else:
        reduction_amount = (monthly_income - income_free_area) * pension_reduction_rate
        pension_after_income_test = max(0.0, basic_pension - reduction_amount)
        return pension_after_income_test

def asset_test(monthly_wealth):
    """
    按资产规模计算养老金削减额（资产测试）。
    :param monthly_wealth: 当前月的金融资产总额。
    :return: 经过资产测试后的每月养老金。
    """
    basic_pension = 2566.36  # 每月基础养老金
    asset_threshold = 321500 / 12  # 对应每月资产，门槛321,500澳元
    pension_reduction_rate = 3 / 1000 # 每1,000澳元减少3澳元

    if monthly_wealth <= asset_threshold:
        return basic_pension
    else:
        reduction_amount = ((monthly_wealth - asset_threshold) / 1000) * pension_reduction_rate
        pension_after_asset_test = max(0.0, basic_pension - reduction_amount)
        return pension_after_asset_test

def run_single_simulation(simulation_seed=None, ltci_coverage_theta=1):
    if simulation_seed is not None:
        np.random.seed(simulation_seed)

    # --- 基本时间设定 ---
    T, dt = 43, 1 / 252
    N = int(T / dt)
    N_per_year = int(1 / dt)         # 约 252
    N_per_month = N_per_year // 12   # 约 21

    # 1) 生成价格与体制路径（日频）
    S_price_path, B_price_path, regimes = simulate_asset_prices(params, T=T, dt=dt)

    # 2) 计算“真·月频”的 EWMA 动态权重和波动率（按月更新→按日展开为常数片段）
    alpha_path, sigma_path = calculate_dynamic_allocations_monthly_true(
        S_price_path, TV=TARGET_VOLATILITY, dt_day=1 / 252, days_per_month=21,
        lambda_ewma=0.8, initial_vol=0.20
    )

    # 3) 日频贴现因子
    discount_factors = compute_discount_factors_from_regimes(regimes, params, dt)
    matrices = pre_computed_matrices

    # 4) 健康状态路径（年频 → 日频）
    # 状态编码：0=健康, 1=轻残, 2=中残, 3=重残, 4=死亡
    health_path_yearly = []
    current_health_state_yearly = 0  # 初始为健康状态
    death_year = T  # 初始化死亡年份为最大值，表示在模拟期内未死亡

    for t_year in range(T):
        health_path_yearly.append(current_health_state_yearly)
        if current_health_state_yearly == 4:
            death_year = t_year
            health_path_yearly.extend([4] * (T - t_year - 1))
            break
        transition_matrix = matrices[t_year]
        prob_vector = transition_matrix[current_health_state_yearly]
        prob_vector = np.maximum(prob_vector, 0.0)
        prob_vector /= np.sum(prob_vector)
        current_health_state_yearly = np.random.choice([0, 1, 2, 3, 4], p=prob_vector)

    health_path_daily = np.repeat(health_path_yearly, N_per_year)[:N]

    # 5) GMDB 投资组合初始化（使用当日的月度α）
    gmdb_total_wealth = GMDB_initial_allocation
    gmdb_stock_wealth = gmdb_total_wealth * alpha_path[0]
    gmdb_bond_wealth = gmdb_total_wealth * (1 - alpha_path[0])

    # 6) 主组合初始化 (这是用于C_annuity_factor_monthly_list的)
    investable_stock_wealth = investable_wealth_initial * alpha_path[0]
    investable_bond_wealth = investable_wealth_initial * (1 - alpha_path[0])

    stock_wealth = investable_stock_wealth
    bond_wealth = investable_bond_wealth

    # ====== 新增：C_age_based_monthly_list 对应的独立主组合初始化 ======
    age_based_stock_wealth = investable_stock_wealth
    age_based_bond_wealth = investable_wealth_initial * (1 - alpha_path[0])
    # =================================================================

    # 记录项（按月）
    C_annuity_factor_monthly_list = []  # 原始年金因子计算的月消费
    C_fixed_monthly_list = []           # 新增：固定月提取 () 策略的月消费
    C_age_based_monthly_list = []       # 新增：按年龄固定比例提取策略的月消费

    S_after_rebal_monthly = []
    B_after_rebal_monthly = []
    alpha_monthly_list = []
    vol_monthly = []
    month_index_list = []
    total_wealth_monthly = []             # 记录主组合月度财富 (C_annuity_factor)
    living_wealth_path_monthly = []       # 存活时的财富 (主组合 C_annuity_factor)

    # ====== 新增：记录 C_age_based_monthly_list 对应的月度财富 ======
    living_wealth_path_age_based_monthly = []
    # ===============================================================

    # === 新增：按月记录未支付健康成本与实际支付的健康成本 ===
    unpaid_health_cost_monthly = []           # 未能支付的健康成本（当月） (针对C_annuity_factor)
    personal_health_cost_paid_monthly = []    # 实际从主组合支付的健康成本（当月） (针对C_annuity_factor)

    # ====== 新增：针对C_age_based_monthly_list的健康成本记录 ======
    unpaid_health_cost_age_based_monthly = []
    personal_health_cost_paid_age_based_monthly = []
    # ===============================================================

    # 新增记录 GMDB 身故给付与期末财富
    gmdb_payout_at_death = 0.0
    terminal_wealth = 0.0

    is_dead = False
    # 纯消费（不含健康成本）
    C_year_annuity_factor = 0.0  # 基于年金因子计算的年度消费
    C_year_age_based = 0.0       # 基于年龄固定比例计算的年度消费

    # 当年健康成本（客户自付部分）以及其“固定月额”
    current_annual_health_cost = 0.0
    customer_actual_health_cost_annual = 0.0
    fixed_monthly_health_cost = 0.0

    # 初始年龄
    start_age = 67
    consecutive_sick_years = 0           # 当前连续生病年份计数
    has_finished_first_sickness_episode = False # 标记是否已经完成过一次生病周期（即生病后康复或转为无费用状态）
    # ====== 新增：Age Pension 相关记录 ======
    age_pension_annuity_factor_path_monthly = []
    age_pension_age_based_path_monthly = []
    age_pension_fixed_path_monthly = []
    # ====================================

    # ===================== 主循环（日频） =====================
    for day in range(N):
        current_year = day // N_per_year
        current_month = day // N_per_month
        current_age = start_age + current_year
        h_t = health_path_daily[day]

        # --- 若已死亡：仅在月初记录，资产保持不变 ---
        if is_dead:
            if day % N_per_month == 0:
                last_total = total_wealth_monthly[-1] if total_wealth_monthly else 0.0
                total_wealth_monthly.append(last_total)
                C_annuity_factor_monthly_list.append(0.0)
                C_fixed_monthly_list.append(0.0)
                C_age_based_monthly_list.append(0.0)
                S_after_rebal_monthly.append(stock_wealth)
                B_after_rebal_monthly.append(bond_wealth)
                alpha_monthly_list.append(0.0)
                vol_monthly.append(0.0)
                month_index_list.append(current_month)
                living_wealth_path_monthly.append(0.0)
                unpaid_health_cost_monthly.append(0.0)
                personal_health_cost_paid_monthly.append(0.0)
                # ====== 新增：age_based 策略死亡后的记录 ======
                living_wealth_path_age_based_monthly.append(0.0)
                unpaid_health_cost_age_based_monthly.append(0.0)
                personal_health_cost_paid_age_based_monthly.append(0.0)
                # ============================================
                # ====== Age Pension 死亡后记录为0 ======
                age_pension_annuity_factor_path_monthly.append(0.0)
                age_pension_age_based_path_monthly.append(0.0)
                age_pension_fixed_path_monthly.append(0.0)
                # ====================================
            continue

        # --- 日度资产增长（主组合与 GMDB） ---
        if day > 0:
            daily_s_return = S_price_path[day] / S_price_path[day - 1]
            daily_b_return = B_price_path[day] / B_price_path[day - 1]
            stock_wealth *= daily_s_return
            bond_wealth *= daily_b_return
            gmdb_stock_wealth *= daily_s_return
            gmdb_bond_wealth *= daily_b_return
            # ====== 新增：age_based 策略的日度资产增长 ======
            age_based_stock_wealth *= daily_s_return
            age_based_bond_wealth *= daily_b_return
            # ============================================

        # --- 当日死亡：触发 GMDB 赔付，合并到遗产（此后不再消费/再平衡/健康支出） ---
        if h_t == 4 and not is_dead:
            is_dead = True
            gmdb_accrued_value = gmdb_stock_wealth + gmdb_bond_wealth
            gmdb_payout_at_death = max(GMDB_initial_allocation, gmdb_accrued_value)

            stock_wealth, bond_wealth = 0.0, 0.0
            gmdb_stock_wealth, gmdb_bond_wealth = 0.0, 0.0
            # ====== 新增：age_based 策略死亡时清零 ======
            age_based_stock_wealth, age_based_bond_wealth = 0.0, 0.0
            # ===========================================

            C_year_annuity_factor = 0.0
            C_year_age_based = 0.0
            current_annual_health_cost = 0.0
            customer_actual_health_cost_annual = 0.0
            fixed_monthly_health_cost = 0.0

            if day % N_per_month == 0:
                total_wealth_monthly.append(0.0)
                C_annuity_factor_monthly_list.append(0.0)
                C_fixed_monthly_list.append(0.0)
                C_age_based_monthly_list.append(0.0)
                S_after_rebal_monthly.append(0.0)
                B_after_rebal_monthly.append(0.0)
                alpha_monthly_list.append(0.0)
                vol_monthly.append(0.0)
                month_index_list.append(current_month)
                living_wealth_path_monthly.append(0.0)
                unpaid_health_cost_monthly.append(0.0)
                personal_health_cost_paid_monthly.append(0.0)
                # ====== 新增：age_based 策略死亡后的记录 ======
                living_wealth_path_age_based_monthly.append(0.0)
                unpaid_health_cost_age_based_monthly.append(0.0)
                personal_health_cost_paid_age_based_monthly.append(0.0)
                # ============================================
                # ====== Age Pension 死亡后记录为0 ======
                age_pension_annuity_factor_path_monthly.append(0.0)
                age_pension_age_based_path_monthly.append(0.0)
                age_pension_fixed_path_monthly.append(0.0)
                # ====================================
            continue

# --- 年初：更新年度消费 C_year 和年度健康成本（若存活） ---
        if day % N_per_year == 0:
            if h_t != 4:
                # ====== 1) 计算“当年总健康成本” (首次生病周期 & 3年限额逻辑) ======
                
                base_cost = HEALTH_COST_BASE_ANNUAL.get(h_t, 0.0)
                is_sick_state = (base_cost > 0) # 状态 1 和 3 为 True

                current_annual_health_cost = 0.0 # 默认归零

                if is_sick_state:
                    # 如果当前是生病状态 (1 或 3)
                    if has_finished_first_sickness_episode:
                        # 之前已经生病过并结束了周期，这是第二次，不再支付
                        current_annual_health_cost = 0.0
                    else:
                        # 处于第一次生病周期中，计数并判断是否在3年内
                        consecutive_sick_years += 1
                        if consecutive_sick_years <= 33:
                            current_annual_health_cost = annual_health_cost(h_t, current_year)
                        else:
                            current_annual_health_cost = 0.0
                else:
                    # 如果当前是无费用状态 (0 或 2)
                    current_annual_health_cost = 0.0
                    # 如果之前在计数（刚结束生病），标记第一次周期已完成
                    if consecutive_sick_years > 0:
                        has_finished_first_sickness_episode = True
                        consecutive_sick_years = 0

                # ====== 2) 客户自付部分 ======
                # 修正：LTCI 覆盖产生费用的状态 (1 和 3)
                if h_t == 1 or h_t == 3:
                    # 保险覆盖 theta 比例，客户支付 (1 - theta)
                    customer_actual_health_cost_annual = (1 - ltci_coverage_theta) * current_annual_health_cost
                else:
                    # 其他状态 (0 或 2)，虽然费用通常为0，但逻辑上是客户自付
                    customer_actual_health_cost_annual = current_annual_health_cost
                
                # 3) 固定为“当年的每月健康支出”
                fixed_monthly_health_cost = customer_actual_health_cost_annual / 12.0

                # 4) 计算策略一：基于年金因子的年度纯消费
                total_wealth_now = (stock_wealth + bond_wealth) # <-- 注意这里是当前策略的财富
                total_wealth_after_health_cost_for_annuity = max(0.0, total_wealth_now - customer_actual_health_cost_annual)
                a_xt = compute_annuity_due_by_survival_discount(current_year, matrices, h_t, discount_factors, dt)
                if a_xt > 1e-12:
                    C_year_annuity_factor = max(0.0, total_wealth_after_health_cost_for_annuity / a_xt)
                else:
                    C_year_annuity_factor = max(0.0, total_wealth_after_health_cost_for_annuity)

                # 5) 计算策略三：基于年龄固定比例的年度纯消费
                # 注意：这里是先扣除健康成本，再按比例提取剩余资产
                withdrawal_rate = 0.0
                if 65 <= current_age <= 74:
                    withdrawal_rate = 0.05
                elif 75 <= current_age <= 79:
                    withdrawal_rate = 0.06
                elif 80 <= current_age <= 84:
                    withdrawal_rate = 0.07
                elif 85 <= current_age <= 89:
                    withdrawal_rate = 0.09
                elif 90 <= current_age <= 95:
                    withdrawal_rate = 0.11
                elif 95 <= current_age <= 100:
                    withdrawal_rate = 0.14
                # ====== 关键：这里要用age_based策略自己的总资产来计算 ======
                age_based_total_wealth_now = (age_based_stock_wealth + age_based_bond_wealth)
                # 年初从总资产中扣除当年健康成本
                wealth_for_age_based_withdrawal = max(0.0, age_based_total_wealth_now - customer_actual_health_cost_annual)
                C_year_age_based = wealth_for_age_based_withdrawal * withdrawal_rate
                # ==========================================================

            else: # If dead at year start
                C_year_annuity_factor = 0.0
                C_year_age_based = 0.0
                current_annual_health_cost = 0.0
                customer_actual_health_cost_annual = 0.0
                fixed_monthly_health_cost = 0.0

        # --- 月度：处理消费和再平衡 ---
        if day % N_per_month == 0:
            # Step 1) 处理健康支出 (三种策略共享同一个健康成本扣减机制，但扣减方式不同)
            # ====== 针对 C_annuity_factor 策略的主组合的健康成本扣除 ======
            total_wealth_annuity = stock_wealth + bond_wealth
            month_health_need = fixed_monthly_health_cost

            # 记录本月实际支付的健康成本和未支付部分
            pay_health_from_main_portfolio_annuity = 0.0
            unpaid_health_current_month_annuity = 0.0

            if month_health_need > 0.0:
                if total_wealth_annuity > 0.0:
                    pay_health_from_main_portfolio_annuity = min(month_health_need, total_wealth_annuity)
                    unpaid_health_current_month_annuity = month_health_need - pay_health_from_main_portfolio_annuity
                    # 按当前持仓比例扣减
                    stock_ratio_annuity = stock_wealth / total_wealth_annuity if total_wealth_annuity > 0 else 0.0
                    bond_ratio_annuity = bond_wealth / total_wealth_annuity if total_wealth_annuity > 0 else 0.0
                    stock_wealth -= pay_health_from_main_portfolio_annuity * stock_ratio_annuity
                    bond_wealth -= pay_health_from_main_portfolio_annuity * bond_ratio_annuity
                else:
                    unpaid_health_current_month_annuity = month_health_need
            personal_health_cost_paid_monthly.append(pay_health_from_main_portfolio_annuity)
            unpaid_health_cost_monthly.append(unpaid_health_current_month_annuity)
            # ===============================================================

            # ====== 新增：针对 C_age_based 策略的独立主组合的健康成本扣除 ======
            total_wealth_age_based = age_based_stock_wealth + age_based_bond_wealth
            # month_health_need 是共享的
            pay_health_from_main_portfolio_age_based = 0.0
            unpaid_health_current_month_age_based = 0.0

            if month_health_need > 0.0:
                if total_wealth_age_based > 0.0:
                    pay_health_from_main_portfolio_age_based = min(month_health_need, total_wealth_age_based)
                    unpaid_health_current_month_age_based = month_health_need - pay_health_from_main_portfolio_age_based
                    stock_ratio_age_based = age_based_stock_wealth / total_wealth_age_based if total_wealth_age_based > 0 else 0.0
                    bond_ratio_age_based = age_based_bond_wealth / total_wealth_age_based if total_wealth_age_based > 0 else 0.0
                    age_based_stock_wealth -= pay_health_from_main_portfolio_age_based * stock_ratio_age_based
                    age_based_bond_wealth -= pay_health_from_main_portfolio_age_based * bond_ratio_age_based
                else:
                    unpaid_health_current_month_age_based = month_health_need
            personal_health_cost_paid_age_based_monthly.append(pay_health_from_main_portfolio_age_based)
            unpaid_health_cost_age_based_monthly.append(unpaid_health_current_month_age_based)
            # ===================================================================


            # Step 2) 计算并记录三种消费策略的月度消费
            # 策略一：原始年金因子法（纯消费部分）
            C_month_annuity_factor = C_year_annuity_factor / 12.0
            C_annuity_factor_monthly_list.append(C_month_annuity_factor)

            # 策略二：固定月提取  - 直接减去月健康成本
            C_month_fixed = 1191.19
            # 固定月提取策略的健康成本直接从这中扣除
            C_month_fixed_after_health = max(0.0, C_month_fixed - month_health_need)
            C_fixed_monthly_list.append(C_month_fixed_after_health)

            # 策略三：按年龄固定比例提取（纯消费部分，健康成本已在年初从总资产中扣除）
            C_month_age_based = C_year_age_based / 12.0
            C_age_based_monthly_list.append(C_month_age_based)

            month_index_list.append(current_month)

            # *** 注意：我们模拟的是原始的“年金因子”策略，所以其总资产变化受其消费和健康成本影响 ***
            # 将原始策略的月度纯消费从主组合 (C_annuity_factor) 中扣除
            total_wealth_annuity = stock_wealth + bond_wealth # 再次更新
            expense_to_deduct_annuity = C_month_annuity_factor

            if total_wealth_annuity > 0 and expense_to_deduct_annuity > 0:
                expense_adj_annuity = min(expense_to_deduct_annuity, total_wealth_annuity)
                stock_ratio_annuity = stock_wealth / total_wealth_annuity if total_wealth_annuity > 0 else 0.0
                bond_ratio_annuity = bond_wealth / total_wealth_annuity if total_wealth_annuity > 0 else 0.0
                stock_wealth -= expense_adj_annuity * stock_ratio_annuity
                bond_wealth -= expense_adj_annuity * bond_ratio_annuity
            elif expense_to_deduct_annuity > 0 and total_wealth_annuity <= 0:
                stock_wealth = 0.0
                bond_wealth = 0.0

            # ====== 新增：将 C_age_based 策略的月度纯消费从独立主组合中扣除 ======
            total_wealth_age_based = age_based_stock_wealth + age_based_bond_wealth # 再次更新
            expense_to_deduct_age_based = C_month_age_based

            if total_wealth_age_based > 0 and expense_to_deduct_age_based > 0:
                expense_adj_age_based = min(expense_to_deduct_age_based, total_wealth_age_based)
                stock_ratio_age_based = age_based_stock_wealth / total_wealth_age_based if total_wealth_age_based > 0 else 0.0
                bond_ratio_age_based = age_based_bond_wealth / total_wealth_age_based if total_wealth_age_based > 0 else 0.0
                age_based_stock_wealth -= expense_adj_age_based * stock_ratio_age_based
                age_based_bond_wealth -= expense_adj_age_based * bond_ratio_age_based
            elif expense_to_deduct_age_based > 0 and total_wealth_age_based <= 0:
                age_based_stock_wealth = 0.0
                age_based_bond_wealth = 0.0
            # =====================================================================

            # Step 4) 计算 Age Pension (从67岁开始领取，对应24个月后)
            current_month_global_index = current_month + current_year * 12 # 从0开始的月份索引
            if current_age >= 67 and not is_dead: # 67岁开始，并且没有死亡
                # 年金因子策略的养老金
                pension_income_test_annuity = income_test(stock_wealth + bond_wealth, is_annuity_income=False)
                pension_asset_test_annuity = asset_test(stock_wealth + bond_wealth)
                age_pension_annuity = min(pension_income_test_annuity, pension_asset_test_annuity)
                age_pension_annuity_factor_path_monthly.append(age_pension_annuity)

                # Age-based 策略的养老金
                pension_income_test_age_based = income_test(age_based_stock_wealth + age_based_bond_wealth, is_annuity_income=False)
                pension_asset_test_age_based = asset_test(age_based_stock_wealth + age_based_bond_wealth)
                age_pension_age_based = min(pension_income_test_age_based, pension_asset_test_age_based)
                age_pension_age_based_path_monthly.append(age_pension_age_based)
                # 固定月提取策略的养老金 (只进行收入测试，不进行资产测试)
                # 注意：这里假设 C_month_fixed_after_health 是实际的“年金收入”，用于收入测试
                pension_income_test_fixed = income_test(0.0, is_annuity_income=True, fixed_monthly_annuity_value=C_month_fixed * 0.6)
                # 对于固定提取策略，养老金直接作为独立收入，不影响现有财富
                age_pension_fixed_path_monthly.append(pension_income_test_fixed) # 固定策略只看收入测试
            else:
                age_pension_annuity_factor_path_monthly.append(0.0)
                age_pension_age_based_path_monthly.append(0.0)
                age_pension_fixed_path_monthly.append(0.0)

            # Step 3) 用“本月的目标波动率权重”再平衡
            alpha_t = alpha_path[day] if (stock_wealth + bond_wealth) > 0 else 0.0 # 使用 annuity 策略的财富判断
            vol_t = sigma_path[day]

            # 主组合 (C_annuity_factor) 再平衡
            current_total_wealth_annuity = stock_wealth + bond_wealth
            if current_total_wealth_annuity > 0:
                stock_wealth = current_total_wealth_annuity * alpha_t
                bond_wealth = current_total_wealth_annuity * (1 - alpha_t)
            else: # 如果资产为负或0，则清零
                stock_wealth = 0.0
                bond_wealth = 0.0

            # GMDB 内部资产再平衡
            gmdb_total_wealth_before_rebal = gmdb_stock_wealth + gmdb_bond_wealth
            if gmdb_total_wealth_before_rebal > 0:
                gmdb_stock_wealth = gmdb_total_wealth_before_rebal * alpha_t
                gmdb_bond_wealth = gmdb_total_wealth_before_rebal * (1 - alpha_t)
            else:
                gmdb_stock_wealth = 0.0
                gmdb_bond_wealth = 0.0

            # ====== 新增：C_age_based 策略的独立主组合再平衡 ======
            current_total_wealth_age_based = age_based_stock_wealth + age_based_bond_wealth
            if current_total_wealth_age_based > 0:
                age_based_stock_wealth = current_total_wealth_age_based * alpha_t # 使用相同的 alpha_t
                age_based_bond_wealth = current_total_wealth_age_based * (1 - alpha_t)
            else:
                age_based_stock_wealth = 0.0
                age_based_bond_wealth = 0.0
            # =====================================================

            # 记录（按月）
            S_after_rebal_monthly.append(stock_wealth)
            B_after_rebal_monthly.append(bond_wealth)
            alpha_monthly_list.append(alpha_t)
            vol_monthly.append(vol_t)
            living_wealth_path_monthly.append(stock_wealth + bond_wealth) # C_annuity_factor 策略的财富
            total_wealth_monthly.append(stock_wealth + bond_wealth)      # 同上

            # ====== 新增：记录 C_age_based 策略的月度财富 ======
            living_wealth_path_age_based_monthly.append(age_based_stock_wealth + age_based_bond_wealth)
            # ====================================================

    # 期末财富（若未死亡则为主组合期末财富；若已死亡则为0）
    terminal_wealth = (stock_wealth + bond_wealth) if not is_dead else 0.0

    return (total_wealth_monthly, C_annuity_factor_monthly_list, S_after_rebal_monthly, B_after_rebal_monthly,
            alpha_monthly_list, vol_monthly, month_index_list, living_wealth_path_monthly,
            gmdb_payout_at_death, terminal_wealth, death_year,
            unpaid_health_cost_monthly, personal_health_cost_paid_monthly, # C_annuity_factor 策略的健康成本
            C_fixed_monthly_list, C_age_based_monthly_list,
            # ====== 新增返回项 ======
            living_wealth_path_age_based_monthly,
            unpaid_health_cost_age_based_monthly,
            personal_health_cost_paid_age_based_monthly,
            age_pension_annuity_factor_path_monthly,
            age_pension_age_based_path_monthly,
            age_pension_fixed_path_monthly)

def configure_initial_wealth(initial_wealth: float):
    """Update W0-dependent globals while keeping the notebook model unchanged."""
    global W0, omega_L, investable_wealth_initial, GMDB_initial_allocation
    W0 = float(initial_wealth)
    if W0 <= 0:
        raise ValueError("W0 must be positive.")
    omega_L = LTCI_PRICE / W0
    if omega_L + omega_G >= 1.0:
        raise ValueError(
            f"W0 is too small: LTCI price A${LTCI_PRICE:,.2f} plus the GMDB allocation leaves no investable wealth."
        )
    investable_wealth_initial = (1.0 - omega_L - omega_G) * W0
    GMDB_initial_allocation = omega_G * W0
    return omega_L


def _pad(values, n):
    arr = np.asarray(values, dtype=float)
    if len(arr) >= n:
        return arr[:n]
    return np.pad(arr, (0, n-len(arr)), constant_values=0.0)


def _conditional_mean_by_year(matrix, years=33):
    """Notebook-compatible mean: average positive monthly values, then average 12 months."""
    out=[]
    for y in range(years):
        vals=matrix[:, y*12:(y+1)*12].reshape(-1)
        positive=vals[vals > 0]
        out.append(float(np.mean(positive)) if positive.size else 0.0)
    return np.asarray(out)


def run_retirement_plan(initial_wealth=407328.0, num_simulations=200, base_seed=2026, years=33):
    """Run the uploaded notebook model and return yearly mean monthly withdrawals and Age Pension."""
    omega = configure_initial_wealth(initial_wealth)
    months = years * 12
    withdrawals=[]; pensions=[]
    age_withdrawals=[]; age_pensions=[]
    fixed_withdrawals=[]; fixed_pensions=[]

    for i in range(int(num_simulations)):
        result = run_single_simulation(simulation_seed=int(base_seed)+i, ltci_coverage_theta=1)
        withdrawals.append(_pad(result[1], months))
        fixed_withdrawals.append(_pad(result[13], months))
        age_withdrawals.append(_pad(result[14], months))
        pensions.append(_pad(result[18], months))
        age_pensions.append(_pad(result[19], months))
        fixed_pensions.append(_pad(result[20], months))

    withdrawals=np.asarray(withdrawals); pensions=np.asarray(pensions)
    age_withdrawals=np.asarray(age_withdrawals); age_pensions=np.asarray(age_pensions)
    fixed_withdrawals=np.asarray(fixed_withdrawals); fixed_pensions=np.asarray(fixed_pensions)

    # Conditional-on-positive means follow the original notebook's aggregation convention.
    s1w=_conditional_mean_by_year(withdrawals, years); s1p=_conditional_mean_by_year(pensions, years)
    s2w=_conditional_mean_by_year(age_withdrawals, years); s2p=_conditional_mean_by_year(age_pensions, years)
    s3w=_conditional_mean_by_year(fixed_withdrawals, years); s3p=_conditional_mean_by_year(fixed_pensions, years)

    rows=[]
    for y in range(years):
        rows.append({
            "Year": y+1, "Age": 67+y,
            "Suggested monthly withdrawal (AUD)": s1w[y],
            "Mean monthly Age Pension (AUD)": s1p[y],
            "Mean total monthly income (AUD)": s1w[y]+s1p[y],
            "Age-based withdrawal (AUD)": s2w[y],
            "Age-based Age Pension (AUD)": s2p[y],
            "Full annuitization withdrawal (AUD)": s3w[y],
            "Full annuitization Age Pension (AUD)": s3p[y],
        })
    return {
        "initial_wealth": W0,
        "ltci_price": LTCI_PRICE,
        "omega_L": omega,
        "gmdb_ratio": omega_G,
        "investable_wealth_initial": investable_wealth_initial,
        "rows": rows,
    }
