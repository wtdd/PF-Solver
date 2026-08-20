#include <random>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <iomanip>
#include <cstdio>
#include <direct.h>
#include "ThreadPool.h"
#include "Structure.h"
#include "nicslu.h"
#include<map>
using namespace std;
string chose_node = "IEEE_118";
#define bus_total 118
#define need 120

int sample_num = 20000, each_injection = 2;
double injec_min = 0.001, injec_max = 0.20;

double theta_eps_min = 1e-6, theta_eps_max = 0.10;
double V_eps_min = 1e-6, V_eps_max = 0.10;

// Training-sample statistics
double train_max_dA = 0.0, train_max_dV = 0.0, train_max_disP = 0.0, train_max_disQ = 0.0;
double cur_max_dA = 0.0;
double cur_max_dV = 0.0;
double cur_max_disP = 0.0;
double cur_max_disQ = 0.0;
double cur_X_signed = 0.0;
double cur_PQ_signed = 0.0;

// Heatmap settings. These values scale the observed training maximum; 2 and 3
// represent twofold and threefold ranges, respectively.
const double SCORE_STEP = 0.2;
const double X_SCORE_MIN = -2.0;
const double X_SCORE_MAX = 2.0;
const double PQ_SCORE_MIN = -2.0;
const double PQ_SCORE_MAX = 2.0;

int each_cell = 10;
int max_try_per_cell = 3000;

const double EPS_DIV = 1e-12;

string path = "D:\\Software\\PF_GenData\\Slover_Gen_Data\\PFdata\\";
string datapath = "D:\\Software\\Python_Code\\Slover\\Data\\";
#define  pool_size  6
#define max_iter 10 // Maximum number of N-R iterations
#define pi 3.141592653589793
// Parallel-execution control
int nxt_step;
HANDLE hEvent;
ThreadPool pool(pool_size);
CRITICAL_SECTION paper;
// Variable definitions
node_Physics np[need];  // Stores node data and adjacent-edge information for concurrent access.
Node node[need];
long double voltage[need], angel[need], voltage_res[need], angel_res[need];
set<int>pq, pv;
int types[need]; // 1:PQ, 2:PV
map<int, int> nomp;
int slack_node = 0;

void load_date() {
	int count = 1;
	string buss = "\\bus.txt";
	string liness = "\\line.txt";
	string gen = "\\gen.txt";
	ifstream inFile;
	std::cout << "Load bus, ";
	inFile.open(path + chose_node + buss, ios::in);
	while (!inFile.eof()) {
		Bus var;
		inFile >> var.bus_number;
		if (var.bus_number == -1) break;
		nomp[var.bus_number] = count;
		var.bus_number = count;
		count++;
		inFile >> var.bus_type >> var.Pd >> var.Qd >> var.Gs >> var.Bs >> var.area >> var.Vm >> var.Va >> var.baseKV >> var.zone >> var.maxVm >> var.minVm;
		np[var.bus_number].bus = var;
		if (var.bus_type == 1) {
			pq.insert(var.bus_number);
			node[var.bus_number].p = var.Pd * -1;
			node[var.bus_number].q = var.Qd * -1;
		}
		else if (var.bus_type == 2) {
			node[var.bus_number].p = var.Pd * -1;
			node[var.bus_number].q = var.Qd * -1;
			node[var.bus_number].v = var.Vm;
		}
		else {
			types[var.bus_number] = 3;
		}
	}
	inFile.close();

	std::cout << "Line  ";
	inFile.open(path + chose_node + liness, ios::in);
	while (!inFile.eof()) {
		Line var;
		inFile >> var.fr;
		if (var.fr == -1) break;
		inFile >> var.to >> var.r >> var.x >> var.b >> var.rateA >> var.rateB >> var.rateC >> var._ratio >> var.angel >> var.status >> var.angmin >> var.angmax;
		var.fr = nomp[var.fr];
		var.to = nomp[var.to];
		//std::cout << var.fr << " " << var.to << endl;
		if (var.status != 0) {
			np[var.fr].lines.push_back(var);
			np[var.to].lines.push_back(var);
		}
	}
	inFile.close();

	std::cout << "Gen ";
	inFile.open(path + chose_node + gen, ios::in);
	while (!inFile.eof()) {
		int var;
		inFile >> var;
		//std::cout << var << endl;
		if (var == -1) break;
		var = nomp[var];

		inFile >> np[var].gendata.Pg >> np[var].gendata.Qg >> np[var].gendata.Qmax >> np[var].gendata.Qmax;
		if (pq.find(var) != pq.end()) {
			node[var].p += np[var].gendata.Pg;
			node[var].q += np[var].gendata.Qg;
		}
		else if (types[var] != 3) {
			pv.insert(var);
			node[var].p += np[var].gendata.Pg;
		}
	}

	inFile.close();
	for (int i = 1; i <= bus_total; i++) {
		if (types[i] == 3) slack_node = i;
		if (types[i] != 3 && pv.find(i) == pv.end()) pq.insert(i);
		node[i].q /= 100;
		node[i].p /= 100;
	}
	for (auto& var : pq) types[var] = 1;
	for (auto& var : pv) types[var] = 2;
	std::cout << " Load Done\n";
}

complex<double> get_complex(double shi, double xu) {
	complex<double> t{ shi,xu };
	return t;
}

void get_Y(int st, int end) {
	//EnterCriticalSection(&paper);
	//cout << nxt_step << endl;
	//LeaveCriticalSection(&paper);
	for (int k = st; k <= end; k++) {
		complex<double> yr{ 0,0 }; //Yr in kk
		yr += (get_complex(np[k].bus.Gs, np[k].bus.Bs) / get_complex(100, 0)); // Divide by 100.0 to obtain the correct admittance matrix.
		//if (np[k].bus.bus_type == 3) {
		//	slack_node = k;// , slack_voltage = np[k].bus.Vm, slack_angel = np[k].bus.Va * pi / 180;
		//}
		for (auto& t : np[k].lines) {
			// Compute Ya.
			long double _ratio = t._ratio;
			long double r = t.r;
			long double x = t.x;

			int to = (t.to == k ? t.fr : t.to);
			// Compute Yr.
			complex<double> b = get_complex(0, t.b) / get_complex(2, 0);
			complex<double> val = get_complex(1, 0) / (get_complex(0, x) + get_complex(r, 0));
			if (_ratio == 0.0) {
				yr += (val + b);
				int flag = 0;
				int len = (int)node[k].Yr.size();  // Merge parallel lines so that Yy is computed correctly.
				for (int i = 0; i < len; i++) {
					if (node[k].Yr[i].first == to) {
						node[k].Yr[i].second = node[k].Yr[i].second - val;
						flag = 1;
					}
				}
				if (flag == 0) node[k].Yr.push_back(make_pair(to, get_complex(0, 0) - val));
			}
			else {
				if (k == t.fr) {
					yr += (val / get_complex(_ratio, 0) + (get_complex(1 - _ratio, 0) * val / get_complex(_ratio * _ratio, 0)) - b);
				}
				else {
					yr += (val / get_complex(_ratio, 0) + (get_complex(_ratio - 1, 0) * val) / get_complex(_ratio, 0) - b);
				}
				int flag = 0;
				int len = (int)node[k].Yr.size();  // Merge parallel lines so that Yy is computed correctly.
				for (int i = 0; i < len; i++) {
					if (node[k].Yr[i].first == to) {
						node[k].Yr[i].second = node[k].Yr[i].second - val / get_complex(_ratio, 0);
						flag = 1;
					}
				}
				if (flag == 0) node[k].Yr.push_back(make_pair(to, get_complex(0, 0) - val / get_complex(_ratio, 0)));
			}
		}
		node[k].Y = yr;
		node[k].Yr.push_back(make_pair(k, yr));
	}
	EnterCriticalSection(&paper);
	nxt_step--;
	if (nxt_step == 0) {
		SetEvent(hEvent); // Signal the event after all threads finish.
		//cout << "Y event" << endl;
	}
	LeaveCriticalSection(&paper);
}
int nonzero = 0;// , nonzeros[need];
std::vector<int> nonzeros(need, 0);
int CSR_x_p[pool_size], CSR_ks[pool_size]; // Track the number of nonzeros preceding each thread's range.
//
int nnz_flag = 1; // Count all nonzeros on the first pass only.

double time_patallel_j_b[pool_size];

void get_J_b(int st, int end, int thrnum, int checks) {
	int total = 0;
	for (int i = st; i <= end; i++) {
		int num = 0;
		long double summin = 0.0, sumadd = 0.0;
		if (types[i] == 3) continue;
		long double p = 0.0, q = 0.0;
		for (auto& var : node[i].Yr) {
			int to = var.first;
			long double Gij = var.second.real();
			long double Bij = var.second.imag();
			long double Theta = angel[i] - angel[to];
			long double adds = voltage[to] * voltage[i] * (Gij * cos(Theta) + Bij * sin(Theta));
			long double mins = voltage[i] * voltage[to] * (Gij * sin(Theta) - Bij * cos(Theta));
			//else fudian_counts = fudian_counts + 11;
			if (to != i) {
				sumadd = sumadd + adds;
				summin = summin + mins;
				if (types[to] != 3) {
					node[i].H[to] = mins;
					num++;
					if (types[to] == 1) {
						node[i].M[to] = adds / voltage[to];
						num++;
					}
					if (types[i] == 1) {
						node[i].K[to] = -1 * adds;
						num++;
						if (types[to] == 1) {
							node[i].L[to] = mins / voltage[to];
							num++;
						}
					}
				}
			}
			else {
				p = adds;
				q = mins;
			}
		}
		node[i].H[i] = summin * -1; num++;
		if (types[i] == 1) {
			node[i].K[i] = sumadd;
			node[i].M[i] = sumadd / voltage[i] + 2 * node[i].Y.real() * voltage[i];
			node[i].L[i] = summin / voltage[i] - 2 * node[i].Y.imag() * voltage[i];
			num += 3;
		}
		node[i].right_hp = sumadd + p;
		node[i].right_hq = summin + q;

		EnterCriticalSection(&paper);
		// cout << "====" << num  << " " << nonzeros[i] << " " << i << endl;
		if (checks == 1) nonzeros[i] = nonzeros[i] + num;
		// cout << "=" << num << " " << nonzeros[i] << " " << i << endl;
		LeaveCriticalSection(&paper);
		total += num;
	}
	EnterCriticalSection(&paper);
	nxt_step--;
	if (nnz_flag) nonzero = nonzero + total;  // Count the nonzero entries in the Jacobian matrix.
	if (nxt_step == 0) SetEvent(hEvent); // Signal the event after all threads finish.
	LeaveCriticalSection(&paper);
}

INicsLU solver = NULL;
_double_t* ax = NULL;
_uint_t* ai = NULL, * ap = NULL;
_double_t* b = NULL, * x = NULL, check_to_end = 1E-6;

map<int, int>mp, mps; // mp stores angle indices and mps stores voltage indices in correction-vector order.

void pre_CSR(int st, int ed, int val) {
	//EnterCriticalSection(&paper[0]);
	//std::cout << st<< " " << ed << " " << val<<endl;
	//LeaveCriticalSection(&paper[0]);
	//stop_watch xianchengjishi;
	//xianchengjishi.restart();
	_uint_t x_p = CSR_x_p[val];
	int ks = CSR_ks[val];
	for (int i = st; i <= ed; i++) {
		if (types[i] == 3) continue;
		ap[ks] = x_p;
		for (auto& var : node[i].H) {
			ax[x_p] = var.second;
			ai[x_p] = mp[var.first];
			x_p++;
		}
		for (auto& var : node[i].M) {
			ax[x_p] = var.second;
			ai[x_p] = mps[var.first];
			x_p++;
		}
		b[ks] = node[i].p - node[i].right_hp;
		ks++;
		if (types[i] == 1) { // Add a K/L row for each PQ bus.
			ap[ks] = x_p;
			for (auto& var : node[i].K) {
				ax[x_p] = var.second;
				//if (mps.find(var.first) == mps.end()) std::cout << "ops" << endl;
				ai[x_p] = mp[var.first];
				x_p++;
			}
			for (auto& var : node[i].L) {
				ax[x_p] = var.second;
				ai[x_p] = mps[var.first];
				x_p++;
			}
			b[ks] = node[i].q - node[i].right_hq;
			ks++;
		}
	}
	//xianchengjishi.stop();
	EnterCriticalSection(&paper);
	//std::cout << val << ": x_p: " << x_p << "ks: " << ks << endl;
	nxt_step--;

	if (nxt_step == 0) {
		SetEvent(hEvent); // Signal the event after all threads finish.
	}
	LeaveCriticalSection(&paper);
}
_uint_t n_p; _double_t* cfg; const _double_t* stat_p; int need_slove;

void init_Slover() {
	//starts.restart();
	int split = bus_total / pool_size, stop = 0;
	int sum = 0, ks = 0;
	for (; stop < pool_size - 1; stop++) {
		for (int j = stop * split + 1; j <= (stop + 1) * split; j++) {
			// cout << "J = " << j << endl;
			sum = sum + nonzeros[j];
			if (types[j] == 1) ks += 2;
			else if (types[j] == 2) ks++;
		}
		CSR_x_p[stop + 1] = sum; // Store the number of nonzeros preceding each thread.
		// Thread 0 starts at 0; thread 1 starts after thread 0's nonzeros.
		// cout << sum << endl;
		CSR_ks[stop + 1] = ks;
	}
	//starts.stop();
	//cout << "sum: " << starts.elapsed_ms() << endl;
	//starts.restart();
	need_slove = (int)pq.size() * 2 + (int)pv.size();

	ax = (_double_t*)malloc(sizeof(_double_t) * nonzero);
	ai = (_uint_t*)malloc(sizeof(_uint_t) * nonzero);
	ap = (_uint_t*)malloc(sizeof(_uint_t) * (need_slove + 1));
	//cout << "Number of nonzeros: " << nonzero << endl;
	n_p = need_slove;
	b = (_double_t*)malloc(sizeof(_double_t) * need_slove);
	x = (_double_t*)malloc(sizeof(_double_t) * need_slove);
	//starts.stop();
	//cout << "malloc: " << starts.elapsed_ms() << endl;
}

// 1: PQ bus; 2: PV bus
int check_converge() {
	int check = 0;
	double var = 0.0, maxx = 0.0;
	for (int i = 1; i <= bus_total; i++) {
		if (types[i] == 3) continue;
		var = std::abs(node[i].p - node[i].right_hp);
		if (maxx < var) maxx = var;
		if (var > check_to_end) check = 1;
		if (types[i] == 1) {
			var = std::abs(node[i].q - node[i].right_hq);
			if (var > check_to_end) check = 1;
			if (maxx < var) maxx = var;
		}
		if (check) break;
	}
	//cout << "  MaxDiff:" << maxx << endl;
	return check;
}

void update_x() {
	int st = 0;
	for (int i = 1; i <= bus_total; i++) {
		if (types[i] == 3) continue;
		angel[i] = angel[i] + *(x + st);
		st++;
		if (types[i] == 1) {
			voltage[i] = voltage[i] + *(x + st);
			st++;
		}
	}
}

double rand_uniform(double l, double r, std::default_random_engine& generator) {
	if (l > r) std::swap(l, r);
	std::uniform_real_distribution<double> dist(l, r);
	return dist(generator);
}

double clean_zero(double x) {
	return std::abs(x) < 1e-12 ? 0.0 : x;
}

std::vector<std::pair<double, double>> make_bins(double min_v, double max_v, double step) {
	std::vector<std::pair<double, double>> bins;

	int kmin = static_cast<int>(std::round(min_v / step));
	int kmax = static_cast<int>(std::round(max_v / step));

	for (int k = kmin; k < kmax; ++k) {
		double low = clean_zero(k * step);
		double high = clean_zero((k + 1) * step);
		bins.push_back({ low, high });
	}

	return bins;
}

double sample_score_in_bin(double low, double high, std::default_random_engine& generator) {
	return rand_uniform(low, high, generator);
}

bool score_in_bin(double x, double low, double high) {
	return x >= low - 1e-9 && x <= high + 1e-9;
}

double dominant_signed_score(
	double signed_a, double denom_a,
	double signed_b, double denom_b
) {
	denom_a = max(std::abs(denom_a), EPS_DIV);
	denom_b = max(std::abs(denom_b), EPS_DIV);

	double score_a = std::abs(signed_a) / denom_a;
	double score_b = std::abs(signed_b) / denom_b;

	if (score_a >= score_b) {
		return signed_a >= 0.0 ? score_a : -score_a;
	}
	else {
		return signed_b >= 0.0 ? score_b : -score_b;
	}
}

void get_signed_max_X(double& signed_dA, double& signed_dV) {
	signed_dA = 0.0;
	signed_dV = 0.0;

	for (int i = 1; i <= bus_total; i++) {
		if (types[i] == 3) continue;

		double dA = static_cast<double>(angel[i] - angel_res[i]);
		if (std::abs(dA) > std::abs(signed_dA)) {
			signed_dA = dA;
		}

		if (types[i] == 1) {
			double dV = static_cast<double>(voltage[i] - voltage_res[i]);
			if (std::abs(dV) > std::abs(signed_dV)) {
				signed_dV = dV;
			}
		}
	}
}

void get_signed_max_PQ(double& signed_disP, double& signed_disQ) {
	signed_disP = 0.0;
	signed_disQ = 0.0;

	for (int i = 1; i <= bus_total; i++) {
		if (types[i] == 3) continue;

		if (std::abs(node[i].ori_p) > EPS_DIV) {
			double relP = static_cast<double>((node[i].p - node[i].ori_p) / std::abs(node[i].ori_p));
			if (std::abs(relP) > std::abs(signed_disP)) {
				signed_disP = relP;
			}
		}

		if (types[i] == 1 && std::abs(node[i].ori_q) > EPS_DIV) {
			double relQ = static_cast<double>((node[i].q - node[i].ori_q) / std::abs(node[i].ori_q));
			if (std::abs(relQ) > std::abs(signed_disQ)) {
				signed_disQ = relQ;
			}
		}
	}
}

void snapshot_current_raw_max(
	double& abs_dA,
	double& abs_dV,
	double& abs_disP,
	double& abs_disQ
) {
	double signed_dA = 0.0, signed_dV = 0.0;
	double signed_disP = 0.0, signed_disQ = 0.0;

	get_signed_max_X(signed_dA, signed_dV);
	get_signed_max_PQ(signed_disP, signed_disQ);

	abs_dA = std::abs(signed_dA);
	abs_dV = std::abs(signed_dV);
	abs_disP = std::abs(signed_disP);
	abs_disQ = std::abs(signed_disQ);
}

void update_train_stats_from_snapshot(
	double abs_dA,
	double abs_dV,
	double abs_disP,
	double abs_disQ
) {
	train_max_dA = max(train_max_dA, abs_dA);
	train_max_dV = max(train_max_dV, abs_dV);
	train_max_disP = max(train_max_disP, abs_disP);
	train_max_disQ = max(train_max_disQ, abs_disQ);
}

void update_current_scores() {
	double signed_dA = 0.0, signed_dV = 0.0;
	double signed_disP = 0.0, signed_disQ = 0.0;

	get_signed_max_X(signed_dA, signed_dV);
	get_signed_max_PQ(signed_disP, signed_disQ);

	cur_max_dA = std::abs(signed_dA);
	cur_max_dV = std::abs(signed_dV);
	cur_max_disP = std::abs(signed_disP);
	cur_max_disQ = std::abs(signed_disQ);

	cur_X_signed = dominant_signed_score(
		signed_dA, train_max_dA,
		signed_dV, train_max_dV
	);

	cur_PQ_signed = dominant_signed_score(
		signed_disP, train_max_disP,
		signed_disQ, train_max_disQ
	);
}
int nonslack_local_id(int bus_id) {
	if (bus_id == slack_node) return -1;
	return bus_id < slack_node ? bus_id - 1 : bus_id - 2;
}

void ensure_dir_exists(const std::string& dir) {
	_mkdir(dir.c_str());
}

template <typename T>
double map_value_or_zero(const std::map<int, T>& mpv, int key) {
	auto it = mpv.find(key);
	if (it == mpv.end()) return 0.0;
	return static_cast<double>(it->second);
}

void calc_bus_power_at_state(int bus_id, const long double* v_state, const long double* a_state, long double& p_calc, long double& q_calc) {
	p_calc = 0.0;
	q_calc = 0.0;
	for (auto& var : node[bus_id].Yr) {
		int to = var.first;
		long double gij = var.second.real();
		long double bij = var.second.imag();
		long double theta = a_state[bus_id] - a_state[to];
		p_calc += v_state[to] * v_state[bus_id] * (gij * cos(theta) + bij * sin(theta));
		q_calc += v_state[bus_id] * v_state[to] * (gij * sin(theta) - bij * cos(theta));
	}
}

void write_common_headers(
	std::ostream& meta_out,
	std::ostream& bus_static_out,
	std::ostream& branch_static_out,
	std::ostream& ybus_out,
	std::ostream& bus_state_out,
	std::ostream& jacobian_out,
	std::ostream& branch_flow_out
) {
	meta_out
		<< "sample_id,split,source,is_pf_from_start_converged,valid_label,nr_iter,"
		<< "x_low,x_high,pq_low,pq_high,x_signed,pq_signed,"
		<< "max_abs_dtheta_start,max_abs_dv_start,max_abs_dp_injection,max_abs_dq_injection,"
		<< "max_abs_dp_start,max_abs_dq_start\n";

	bus_static_out
		<< "bus_id,bus0,nonslack0,type,is_slack,is_pv,is_pq,"
		<< "base_kv,vm_init,va_init_rad,va_init_deg,vmax,vmin,"
		<< "pd_mw,qd_mvar,gs,bs,p_spec0_pu,q_spec0_pu\n";

	branch_static_out
		<< "branch_id,from_bus,to_bus,from_bus0,to_bus0,r,x,b,rateA,rateB,rateC,tap,shift_deg,status,angmin,angmax\n";

	ybus_out << "from_bus,to_bus,from_bus0,to_bus0,g,b,is_diagonal\n";

	bus_state_out
		<< "sample_id,split,bus_id,bus0,nonslack0,type,is_slack,is_pv,is_pq,"
		<< "p_spec_pu,q_spec_pu,"
		<< "vm_start,va_start_rad,va_start_deg,p_calc_start,q_calc_start,dp_start,dq_start,"
		<< "vm_true,va_true_rad,va_true_deg,p_calc_true,q_calc_true,dp_true,dq_true,"
		<< "dtheta_label,dv_label,u_start_re,u_start_im,u_true_re,u_true_im,s_spec_re,s_spec_im\n";

	jacobian_out
		<< "sample_id,split,from_bus,to_bus,from_bus0,to_bus0,from_nonslack0,to_nonslack0,"
		<< "g,b,H,M,K,L,is_diagonal\n";

	branch_flow_out
		<< "sample_id,split,state,from_bus,to_bus,from_bus0,to_bus0,g,b,p_contrib,q_contrib\n";
}

void write_common_static(
	std::ostream& bus_static_out,
	std::ostream& branch_static_out,
	std::ostream& ybus_out
) {
	bus_static_out << std::fixed << std::setprecision(15);
	for (int i = 1; i <= bus_total; i++) {
		bus_static_out
			<< i << "," << i - 1 << "," << nonslack_local_id(i) << ","
			<< types[i] << "," << (types[i] == 3) << "," << (types[i] == 2) << "," << (types[i] == 1) << ","
			<< np[i].bus.baseKV << "," << np[i].bus.Vm << "," << np[i].bus.Va * pi / 180.0 << "," << np[i].bus.Va << ","
			<< np[i].bus.maxVm << "," << np[i].bus.minVm << ","
			<< np[i].bus.Pd << "," << np[i].bus.Qd << "," << np[i].bus.Gs << "," << np[i].bus.Bs << ","
			<< node[i].ori_p << "," << node[i].ori_q << "\n";
	}

	branch_static_out << std::fixed << std::setprecision(15);
	int branch_id = 0;
	for (int i = 1; i <= bus_total; i++) {
		for (auto& var : np[i].lines) {
			if (i != var.fr) continue;
			branch_static_out
				<< branch_id++ << ","
				<< var.fr << "," << var.to << "," << var.fr - 1 << "," << var.to - 1 << ","
				<< var.r << "," << var.x << "," << var.b << ","
				<< var.rateA << "," << var.rateB << "," << var.rateC << ","
				<< var._ratio << "," << var.angel << "," << var.status << ","
				<< var.angmin << "," << var.angmax << "\n";
		}
	}

	ybus_out << std::fixed << std::setprecision(15);
	for (int i = 1; i <= bus_total; i++) {
		for (auto& var : node[i].Yr) {
			int to = var.first;
			ybus_out
				<< i << "," << to << "," << i - 1 << "," << to - 1 << ","
				<< var.second.real() << "," << var.second.imag() << "," << (i == to) << "\n";
		}
	}
}

void write_common_branch_flow_rows(
	int sample_id,
	const std::string& split,
	const std::string& state_name,
	const long double* v_state,
	const long double* a_state,
	std::ostream& branch_flow_out
) {
	branch_flow_out << std::fixed << std::setprecision(15);
	for (int i = 1; i <= bus_total; i++) {
		for (auto& var : node[i].Yr) {
			int to = var.first;
			if (to == i) continue;
			long double gij = var.second.real();
			long double bij = var.second.imag();
			long double theta = a_state[i] - a_state[to];
			long double p = v_state[to] * v_state[i] * (gij * cos(theta) + bij * sin(theta));
			long double q = v_state[i] * v_state[to] * (gij * sin(theta) - bij * cos(theta));
			branch_flow_out
				<< sample_id << "," << split << "," << state_name << ","
				<< i << "," << to << "," << i - 1 << "," << to - 1 << ","
				<< gij << "," << bij << "," << p << "," << q << "\n";
		}
	}
}

void write_common_sample(
	int sample_id,
	const std::string& split,
	const std::string& source,
	bool is_pf_from_start_converged,
	bool valid_label,
	int nr_iter,
	std::ostream& meta_out,
	std::ostream& bus_state_out,
	std::ostream& jacobian_out,
	std::ostream& branch_flow_out,
	double x_low = 0.0,
	double x_high = 0.0,
	double pq_low = 0.0,
	double pq_high = 0.0
) {
	double abs_dA = 0.0, abs_dV = 0.0, abs_disP = 0.0, abs_disQ = 0.0;
	snapshot_current_raw_max(abs_dA, abs_dV, abs_disP, abs_disQ);

	long double max_dp_start = 0.0, max_dq_start = 0.0;
	for (int i = 1; i <= bus_total; i++) {
		if (types[i] == 3) continue;
		long double dp = std::abs(node[i].p - node[i].right_hp);
		if (dp > max_dp_start) max_dp_start = dp;
		if (types[i] == 1) {
			long double dq = std::abs(node[i].q - node[i].right_hq);
			if (dq > max_dq_start) max_dq_start = dq;
		}
	}

	meta_out << std::fixed << std::setprecision(15)
		<< sample_id << "," << split << "," << source << ","
		<< (is_pf_from_start_converged ? 1 : 0) << "," << (valid_label ? 1 : 0) << "," << nr_iter << ","
		<< x_low << "," << x_high << "," << pq_low << "," << pq_high << ","
		<< cur_X_signed << "," << cur_PQ_signed << ","
		<< abs_dA << "," << abs_dV << "," << abs_disP << "," << abs_disQ << ","
		<< max_dp_start << "," << max_dq_start << "\n";

	bus_state_out << std::fixed << std::setprecision(15);
	for (int i = 1; i <= bus_total; i++) {
		long double p_true = 0.0, q_true = 0.0;
		calc_bus_power_at_state(i, voltage_res, angel_res, p_true, q_true);

		long double p_spec = node[i].p;
		long double q_spec = node[i].q;
		long double dp_start = p_spec - node[i].right_hp;
		long double dq_start = q_spec - node[i].right_hq;
		long double dp_true = p_spec - p_true;
		long double dq_true = q_spec - q_true;

		if (types[i] == 3) {
			dp_start = 0.0;
			dq_start = 0.0;
			dp_true = 0.0;
			dq_true = 0.0;
		}
		else if (types[i] == 2) {
			dq_start = 0.0;
			dq_true = 0.0;
		}

		long double dtheta = (types[i] == 3) ? 0.0 : angel_res[i] - angel[i];
		long double dv = (types[i] == 1) ? voltage_res[i] - voltage[i] : 0.0;
		long double u_start_re = voltage[i] * cos(angel[i]);
		long double u_start_im = voltage[i] * sin(angel[i]);
		long double u_true_re = voltage_res[i] * cos(angel_res[i]);
		long double u_true_im = voltage_res[i] * sin(angel_res[i]);

		bus_state_out
			<< sample_id << "," << split << ","
			<< i << "," << i - 1 << "," << nonslack_local_id(i) << ","
			<< types[i] << "," << (types[i] == 3) << "," << (types[i] == 2) << "," << (types[i] == 1) << ","
			<< p_spec << "," << q_spec << ","
			<< voltage[i] << "," << angel[i] << "," << angel[i] * 180.0 / pi << ","
			<< node[i].right_hp << "," << node[i].right_hq << "," << dp_start << "," << dq_start << ","
			<< voltage_res[i] << "," << angel_res[i] << "," << angel_res[i] * 180.0 / pi << ","
			<< p_true << "," << q_true << "," << dp_true << "," << dq_true << ","
			<< dtheta << "," << dv << ","
			<< u_start_re << "," << u_start_im << "," << u_true_re << "," << u_true_im << ","
			<< p_spec << "," << q_spec << "\n";
	}

	jacobian_out << std::fixed << std::setprecision(15);
	for (int i = 1; i <= bus_total; i++) {
		if (types[i] == 3) continue;
		for (auto& var : node[i].Yr) {
			int to = var.first;
			jacobian_out
				<< sample_id << "," << split << ","
				<< i << "," << to << "," << i - 1 << "," << to - 1 << ","
				<< nonslack_local_id(i) << "," << nonslack_local_id(to) << ","
				<< var.second.real() << "," << var.second.imag() << ","
				<< map_value_or_zero(node[i].H, to) << ","
				<< map_value_or_zero(node[i].M, to) << ","
				<< map_value_or_zero(node[i].K, to) << ","
				<< map_value_or_zero(node[i].L, to) << ","
				<< (i == to) << "\n";
		}
	}

	write_common_branch_flow_rows(sample_id, split, "start", voltage, angel, branch_flow_out);
	write_common_branch_flow_rows(sample_id, split, "true", voltage_res, angel_res, branch_flow_out);
}

void write_common_invalid_sample(
	int sample_id,
	const std::string& split,
	const std::string& source,
	std::ostream& meta_out,
	std::ostream& bus_state_out,
	std::ostream& jacobian_out,
	std::ostream& branch_flow_out
) {
	double signed_disP = 0.0, signed_disQ = 0.0;
	get_signed_max_PQ(signed_disP, signed_disQ);
	double abs_disP = std::abs(signed_disP);
	double abs_disQ = std::abs(signed_disQ);

	long double max_dp_start = 0.0, max_dq_start = 0.0;
	for (int i = 1; i <= bus_total; i++) {
		if (types[i] == 3) continue;
		long double dp = std::abs(node[i].p - node[i].right_hp);
		if (dp > max_dp_start) max_dp_start = dp;
		if (types[i] == 1) {
			long double dq = std::abs(node[i].q - node[i].right_hq);
			if (dq > max_dq_start) max_dq_start = dq;
		}
	}

	meta_out << std::fixed << std::setprecision(15)
		<< sample_id << "," << split << "," << source << ","
		<< 0 << "," << 0 << "," << -1 << ","
		<< 0.0 << "," << 0.0 << "," << 0.0 << "," << 0.0 << ","
		<< 0.0 << "," << 0.0 << ","
		<< 0.0 << "," << 0.0 << "," << abs_disP << "," << abs_disQ << ","
		<< max_dp_start << "," << max_dq_start << "\n";

	bus_state_out << std::fixed << std::setprecision(15);
	for (int i = 1; i <= bus_total; i++) {
		long double p_spec = node[i].p;
		long double q_spec = node[i].q;
		long double dp_start = p_spec - node[i].right_hp;
		long double dq_start = q_spec - node[i].right_hq;

		if (types[i] == 3) {
			dp_start = 0.0;
			dq_start = 0.0;
		}
		else if (types[i] == 2) {
			dq_start = 0.0;
		}

		long double u_start_re = voltage[i] * cos(angel[i]);
		long double u_start_im = voltage[i] * sin(angel[i]);

		bus_state_out
			<< sample_id << "," << split << ","
			<< i << "," << i - 1 << "," << nonslack_local_id(i) << ","
			<< types[i] << "," << (types[i] == 3) << "," << (types[i] == 2) << "," << (types[i] == 1) << ","
			<< p_spec << "," << q_spec << ","
			<< voltage[i] << "," << angel[i] << "," << angel[i] * 180.0 / pi << ","
			<< node[i].right_hp << "," << node[i].right_hq << "," << dp_start << "," << dq_start << ","
			<< 0.0 << "," << 0.0 << "," << 0.0 << ","
			<< 0.0 << "," << 0.0 << "," << 0.0 << "," << 0.0 << ","
			<< 0.0 << "," << 0.0 << ","
			<< u_start_re << "," << u_start_im << "," << 0.0 << "," << 0.0 << ","
			<< p_spec << "," << q_spec << "\n";
	}

	jacobian_out << std::fixed << std::setprecision(15);
	for (int i = 1; i <= bus_total; i++) {
		if (types[i] == 3) continue;
		for (auto& var : node[i].Yr) {
			int to = var.first;
			jacobian_out
				<< sample_id << "," << split << ","
				<< i << "," << to << "," << i - 1 << "," << to - 1 << ","
				<< nonslack_local_id(i) << "," << nonslack_local_id(to) << ","
				<< var.second.real() << "," << var.second.imag() << ","
				<< map_value_or_zero(node[i].H, to) << ","
				<< map_value_or_zero(node[i].M, to) << ","
				<< map_value_or_zero(node[i].K, to) << ","
				<< map_value_or_zero(node[i].L, to) << ","
				<< (i == to) << "\n";
		}
	}

	write_common_branch_flow_rows(sample_id, split, "start", voltage, angel, branch_flow_out);

	branch_flow_out << std::fixed << std::setprecision(15);
	for (int i = 1; i <= bus_total; i++) {
		for (auto& var : node[i].Yr) {
			int to = var.first;
			if (to == i) continue;
			branch_flow_out
				<< sample_id << "," << split << "," << "true" << ","
				<< i << "," << to << "," << i - 1 << "," << to - 1 << ","
				<< var.second.real() << "," << var.second.imag() << ","
				<< 0.0 << "," << 0.0 << "\n";
		}
	}
}

void re_init_train(std::default_random_engine& generator) {
	std::uniform_real_distribution<double> dist_log_theta(std::log(theta_eps_min), std::log(theta_eps_max));
	std::uniform_real_distribution<double> dist_log_V(std::log(V_eps_min), std::log(V_eps_max));
	std::uniform_real_distribution<double> dist_sign(-1.0, 1.0);

	double theta_eps = std::exp(dist_log_theta(generator));
	double V_eps = std::exp(dist_log_V(generator));

	for (int i = 1; i <= bus_total; i++) {
		if (types[i] == 3) continue;

		double add_angel = dist_sign(generator) * theta_eps;
		angel[i] = angel_res[i] + add_angel;

		if (types[i] == 1) {
			double add_voltage = dist_sign(generator) * V_eps;
			voltage[i] = voltage_res[i] + add_voltage;
		}
		else if (types[i] == 2) {
			voltage[i] = voltage_res[i];
		}
	}
}

void set_PQ_train(std::default_random_engine& generator) {
	std::uniform_real_distribution<double> dist_sign(0.0, 1.0);
	std::uniform_real_distribution<double> dist_value(injec_min, injec_max);

	for (int i = 1; i <= bus_total; i++) {
		if (types[i] == 3) continue;

		double sign = dist_sign(generator) < 0.5 ? -1.0 : 1.0;
		double relP = sign * dist_value(generator);
		node[i].p = node[i].ori_p + relP * std::abs(node[i].ori_p);

		if (types[i] == 1) {
			sign = dist_sign(generator) < 0.5 ? -1.0 : 1.0;
			double relQ = sign * dist_value(generator);
			node[i].q = node[i].ori_q + relQ * std::abs(node[i].ori_q);
		}
	}
}

void set_PQ_heatmap(double target_PQ_signed, std::default_random_engine& generator) {
	double abs_score = std::abs(target_PQ_signed);
	double sign = target_PQ_signed >= 0.0 ? 1.0 : -1.0;

	std::vector<int> p_candidates;
	std::vector<int> q_candidates;

	for (int i = 1; i <= bus_total; i++) {
		if (types[i] == 3) continue;

		if (std::abs(node[i].ori_p) > EPS_DIV) {
			p_candidates.push_back(i);
		}

		if (types[i] == 1 && std::abs(node[i].ori_q) > EPS_DIV) {
			q_candidates.push_back(i);
		}
	}

	double small_score = 0.5 * abs_score;

	for (int i = 1; i <= bus_total; i++) {
		if (types[i] == 3) continue;

		double relP = rand_uniform(-small_score, small_score, generator) * train_max_disP;
		node[i].p = node[i].ori_p + relP * std::abs(node[i].ori_p);

		if (types[i] == 1) {
			double relQ = rand_uniform(-small_score, small_score, generator) * train_max_disQ;
			node[i].q = node[i].ori_q + relQ * std::abs(node[i].ori_q);
		}
	}

	// Force one dominant P or Q perturbation so that PQ_signed falls in the target bin.
	bool use_Q = (!q_candidates.empty() && rand_uniform(0.0, 1.0, generator) < 0.5);

	if (use_Q) {
		int idx = static_cast<int>(rand_uniform(0.0, static_cast<double>(q_candidates.size()), generator));
		if (idx >= static_cast<int>(q_candidates.size())) {
			idx = static_cast<int>(q_candidates.size()) - 1;
		}

		int bus = q_candidates[idx];
		double relQ = sign * abs_score * train_max_disQ;
		node[bus].q = node[bus].ori_q + relQ * std::abs(node[bus].ori_q);
	}
	else {
		int idx = static_cast<int>(rand_uniform(0.0, static_cast<double>(p_candidates.size()), generator));
		if (idx >= static_cast<int>(p_candidates.size())) {
			idx = static_cast<int>(p_candidates.size()) - 1;
		}

		int bus = p_candidates[idx];
		double relP = sign * abs_score * train_max_disP;
		node[bus].p = node[bus].ori_p + relP * std::abs(node[bus].ori_p);
	}
}

void re_init_heatmap(double target_X_signed, std::default_random_engine& generator) {
	double abs_score = std::abs(target_X_signed);
	double sign = target_X_signed >= 0.0 ? 1.0 : -1.0;

	std::vector<int> angle_candidates;
	std::vector<int> voltage_candidates;

	for (int i = 1; i <= bus_total; i++) {
		if (types[i] == 3) continue;

		angle_candidates.push_back(i);

		if (types[i] == 1) {
			voltage_candidates.push_back(i);
		}
	}

	// Apply a small perturbation to every bus so the sample does not vary at only one bus.
	double small_score = 0.5 * abs_score;

	for (int i = 1; i <= bus_total; i++) {
		if (types[i] == 3) continue;

		double addA = rand_uniform(-small_score, small_score, generator) * train_max_dA;
		angel[i] = angel_res[i] + addA;

		if (types[i] == 1) {
			double addV = rand_uniform(-small_score, small_score, generator) * train_max_dV;
			voltage[i] = voltage_res[i] + addV;
		}
		else if (types[i] == 2) {
			voltage[i] = voltage_res[i];
		}
	}

	// Force one dominant X perturbation so that X_signed falls in the target bin.
	// Prefer angle perturbations in most cases to reduce voltage-limit violations.
	bool use_V = (!voltage_candidates.empty() && rand_uniform(0.0, 1.0, generator) < 0.3);

	if (use_V) {
		int idx = static_cast<int>(rand_uniform(0.0, static_cast<double>(voltage_candidates.size()), generator));
		if (idx >= static_cast<int>(voltage_candidates.size())) {
			idx = static_cast<int>(voltage_candidates.size()) - 1;
		}

		int bus = voltage_candidates[idx];
		voltage[bus] = voltage_res[bus] + sign * abs_score * train_max_dV;
	}
	else {
		int idx = static_cast<int>(rand_uniform(0.0, static_cast<double>(angle_candidates.size()), generator));
		if (idx >= static_cast<int>(angle_candidates.size())) {
			idx = static_cast<int>(angle_candidates.size()) - 1;
		}

		int bus = angle_candidates[idx];
		angel[bus] = angel_res[bus] + sign * abs_score * train_max_dA;
	}
}

void re_init(int is_test, std::default_random_engine& generator) {
	std::uniform_real_distribution<double> dist_log_theta(std::log(theta_eps_min), std::log(theta_eps_max));
	std::uniform_real_distribution<double> dist_log_V(std::log(V_eps_min), std::log(V_eps_max));
	std::uniform_real_distribution<double> dist_sign(-1.0, 1.0);
	// Use one epsilon scale for the graph; sample each bus independently within [-epsilon, +epsilon].
	double theta_eps = std::exp(dist_log_theta(generator));
	double V_eps = std::exp(dist_log_V(generator));
	for (int i = 1; i <= bus_total; i++) {
		if (types[i] == 3) continue;
		double add_angel = dist_sign(generator) * theta_eps;
		if (abs(add_angel) > train_max_dA) train_max_dA = abs(add_angel);
		angel[i] = angel_res[i] + add_angel;
		if (types[i] == 1) {
			double add_voltage = dist_sign(generator) * V_eps;
			if (abs(add_voltage) > train_max_dV) train_max_dV = abs(add_voltage);
			voltage[i] = voltage_res[i] + add_voltage;
			// Previously generated voltages with upper and lower bounds of 1.10 and 0.90.
			//double safe_upper = min(V_eps, 1.10 - voltage_res[i]);
			//double safe_lower = max(-V_eps, 0.90 - voltage_res[i]);
			//if (safe_lower > safe_upper) std::swap(safe_lower, safe_upper);
			//std::uniform_real_distribution<double> dist_V_safe(safe_lower, safe_upper);
			//double add_voltage = dist_V_safe(generator);


		}
		else if (types[i] == 2) voltage[i] = voltage_res[i];
	}
}

int main() {
	int cnt_small = 0, cnt_mid = 0, cnt_large = 0;
	std::cout.setf(std::ios::scientific); std::cout.precision(2);
	hEvent = CreateEvent(NULL, TRUE, FALSE, NULL); // Initially nonsignaled.
	InitializeCriticalSection(&paper);
	load_date();
	int stop = 0; int split = bus_total / pool_size; nxt_step = pool_size;
	for (; stop < pool_size - 1; stop++) { pool.enqueue([stop, split] { get_Y(stop * split + 1, (stop + 1) * split); }); }
	pool.enqueue([stop, split] { get_Y(stop * split + 1, bus_total); });
	WaitForSingleObject(hEvent, INFINITE);	ResetEvent(hEvent);
	std::default_random_engine generator(static_cast<unsigned>(std::time(nullptr)));
	for (int i = 1; i <= bus_total; i++) node[i].ori_p = node[i].p, node[i].ori_q = node[i].q;
	ensure_dir_exists(datapath + chose_node);
	std::string common_dir = datapath + chose_node;
	ensure_dir_exists(common_dir);
	const std::string commonFileNames[] = {
		"meta.csv",
		"bus_static.csv",
		"branch_static.csv",
		"ybus.csv",
		"bus_state.csv",
		"jacobian_start.csv",
		"branch_flow.csv"
	};
	for (const auto& name : commonFileNames) std::remove((common_dir + "\\" + name).c_str());
	std::ofstream common_meta_out(common_dir + "\\meta.csv", std::ios::app);
	std::ofstream common_bus_static_out(common_dir + "\\bus_static.csv", std::ios::app);
	std::ofstream common_branch_static_out(common_dir + "\\branch_static.csv", std::ios::app);
	std::ofstream common_ybus_out(common_dir + "\\ybus.csv", std::ios::app);
	std::ofstream common_bus_state_out(common_dir + "\\bus_state.csv", std::ios::app);
	std::ofstream common_jacobian_out(common_dir + "\\jacobian_start.csv", std::ios::app);
	std::ofstream common_branch_flow_out(common_dir + "\\branch_flow.csv", std::ios::app);
	write_common_headers(common_meta_out, common_bus_static_out, common_branch_static_out, common_ybus_out, common_bus_state_out, common_jacobian_out, common_branch_flow_out);
	write_common_static(common_bus_static_out, common_branch_static_out, common_ybus_out);
	int solver_check = 0, total_sample = 0, is_test = 0;
	int common_sample_id = 0;

	for (int tt = 1; tt <= sample_num; tt++) {
		double sign = 0.0, delta = 0.0, factor = 0.0;
		if (tt > sample_num) is_test = 1; // test = 1 selects test-set output.
		if (tt == sample_num + 1) cout << "****************************************************" << endl;
		// Add perturbations to P and Q.
		set_PQ_train(generator);
		nonzero = 0; nnz_flag = 1;
		for (int i = 1; i <= bus_total; i++) { nonzeros[i] = 0, voltage[i] = np[i].bus.Vm;	angel[i] = np[i].bus.Va * pi / 180; }
		stop = 0, nxt_step = pool_size;
		for (; stop < pool_size - 1; stop++) { pool.enqueue([stop, split] { get_J_b(stop * split + 1, (stop + 1) * split, stop, 1); }); }
		pool.enqueue([stop, split] { get_J_b(stop * split + 1, bus_total, stop, 1); });
		WaitForSingleObject(hEvent, INFINITE); ResetEvent(hEvent);
		nnz_flag = 0; // The first J_b pass determines the total nonzero count; later passes do not add to it.
		int res = check_converge();	int val = 0; mp.clear(); mps.clear();
		for (int i = 1; i <= bus_total; i++) {
			if (types[i] == 3) continue;
			mp[i] = val; val++;
			if (types[i] == 1) { mps[i] = val; val++; }
		}
		init_Slover(); stop = 0, nxt_step = pool_size;
		for (; stop < pool_size - 1; stop++) { pool.enqueue([stop, split] { pre_CSR(stop * split + 1, (stop + 1) * split, stop); }); }
		pool.enqueue([stop, split] { pre_CSR(stop * split + 1, bus_total, stop); });
		WaitForSingleObject(hEvent, INFINITE); ResetEvent(hEvent);
		ap[need_slove] = (_uint_t)nonzero; NicsLU_Initialize(&solver, &cfg, &stat_p, NULL);
		cfg[0] = 1.; cfg[3] = 4;
		solver_check = solver->Analyze(n_p, ax, ai, ap, MATRIX_REAL_ROW, NULL, NULL, NULL, NULL); solver_check = solver->CreateThreads(pool_size); solver_check = solver->FactorizeMatrix(ax, pool_size); solver_check = solver->Solve(b, x);
		if (solver_check != 0) { std::cout << "ops1"; return 0; }
		update_x();
		int step = 1;
		while (step <= max_iter) {
			stop = 0, nxt_step = pool_size;
			for (; stop < pool_size - 1; stop++) { pool.enqueue([stop, split] { get_J_b(stop * split + 1, (stop + 1) * split, stop, 2); }); }
			pool.enqueue([stop, split] { get_J_b(stop * split + 1, bus_total, stop, 2); });
			WaitForSingleObject(hEvent, INFINITE); ResetEvent(hEvent);
			if (check_converge() == 0) break;
			stop = 0, nxt_step = pool_size;
			for (; stop < pool_size - 1; stop++) { pool.enqueue([stop, split] { pre_CSR(stop * split + 1, (stop + 1) * split, stop); }); }
			pool.enqueue([stop, split] { pre_CSR(stop * split + 1, bus_total, stop); });
			WaitForSingleObject(hEvent, INFINITE); ResetEvent(hEvent);
			solver_check = solver->FactorizeMatrix(ax, 0); solver_check = solver->Solve(b, x);
			if (solver_check != 0) { std::cout << "ops2"; return 0; }
			update_x();	step++;
		}

		for (int i = 1; i <= bus_total; i++) angel_res[i] = angel[i], voltage_res[i] = voltage[i];
		//std::cout << "Iter： " << step << endl;
		if (step < max_iter) {
			// With the converged angle and voltage, construct x_init = x_final + epsilon,
			// where epsilon is sampled from N(-delta, +delta).
			for (int i = 1; i <= each_injection; i++) {
				re_init_train(generator);// Generate a perturbed x_init from the converged solution.
				double cand_abs_dA = 0.0, cand_abs_dV = 0.0;
				double cand_abs_disP = 0.0, cand_abs_disQ = 0.0;

				snapshot_current_raw_max(
					cand_abs_dA,
					cand_abs_dV,
					cand_abs_disP,
					cand_abs_disQ
				);
				std::ostringstream common_meta_buf, common_bus_state_buf, common_jacobian_buf, common_branch_flow_buf;
				int this_common_sample_id = common_sample_id++;
				int iter_num = 0;
				bool sample_converged = false;
				while (iter_num < max_iter) {
					stop = 0, nxt_step = pool_size;
					for (; stop < pool_size - 1; stop++) { pool.enqueue([stop, split] { get_J_b(stop * split + 1, (stop + 1) * split, stop, 2); }); }
					pool.enqueue([stop, split] { get_J_b(stop * split + 1, bus_total, stop, 2); });
					WaitForSingleObject(hEvent, INFINITE); ResetEvent(hEvent);

					if (iter_num == 0) {
						write_common_sample(
							this_common_sample_id,
							"train",
							"nr_verified_start",
							true,
							true,
							-1,
							common_meta_buf,
							common_bus_state_buf,
							common_jacobian_buf,
							common_branch_flow_buf
						);
					}

					if (check_converge() == 0) {
						sample_converged = true;
						break;
					}
					stop = 0, nxt_step = pool_size;
					for (; stop < pool_size - 1; stop++) { pool.enqueue([stop, split] { pre_CSR(stop * split + 1, (stop + 1) * split, stop); }); }
					pool.enqueue([stop, split] { pre_CSR(stop * split + 1, bus_total, stop); });
					WaitForSingleObject(hEvent, INFINITE); ResetEvent(hEvent);
					solver_check = solver->FactorizeMatrix(ax, 0); solver_check = solver->Solve(b, x);
					if (solver_check != 0) { std::cout << "ops2"; return 0; }
					update_x();
					iter_num++;
				}
				//cout << "iter_num:" << iter_num << endl << endl;
				if (sample_converged) {
					update_train_stats_from_snapshot(
						cand_abs_dA,
						cand_abs_dV,
						cand_abs_disP,
						cand_abs_disQ
					);

					common_meta_out << common_meta_buf.str();
					common_bus_state_out << common_bus_state_buf.str();
					common_jacobian_out << common_jacobian_buf.str();
					common_branch_flow_out << common_branch_flow_buf.str();

					total_sample++;
					if (total_sample % 500 == 0) cout << "Generated training samples: " << total_sample << endl;
				}
				else {
					cout << "Train sample error" << endl;
				}
			}
		}
		else std::cout << "PQ当前噪音不收敛" << endl;
		//cout << "  " << endl;
		free(ax); free(ai); free(ap); solver->DestroyThreads(); solver->Free(); solver = nullptr;
	}
	std::cout << "构造训练样本数：" << total_sample << endl;
	// ================================================================
	// Stage 2: Generate heatmap test data after train data is finished
	// ================================================================

	// Prevent division by zero.
	train_max_dA = max(train_max_dA, EPS_DIV);
	train_max_dV = max(train_max_dV, EPS_DIV);
	train_max_disP = max(train_max_disP, EPS_DIV);
	train_max_disQ = max(train_max_disQ, EPS_DIV);

	cout << "****************************************************" << endl;
	cout << "Train boundary stats:" << endl;
	cout << "train_max_dA   = " << train_max_dA << endl;
	cout << "train_max_dV   = " << train_max_dV << endl;
	cout << "train_max_disP = " << train_max_disP << endl;
	cout << "train_max_disQ = " << train_max_disQ << endl;
	cout << "Begin heatmap test generation..." << endl;
	cout << "****************************************************" << endl;


	// ---------------------------------------------------------------
	// Release solver memory used by the current candidate.
	// ---------------------------------------------------------------
	auto release_solver_memory = [&]() {
		if (ax != nullptr) { free(ax); ax = nullptr; }
		if (ai != nullptr) { free(ai); ai = nullptr; }
		if (ap != nullptr) { free(ap); ap = nullptr; }
		if (b != nullptr) { free(b);  b = nullptr; }
		if (x != nullptr) { free(x);  x = nullptr; }

		if (solver != nullptr) {
			solver->DestroyThreads();
			solver->Free();
			solver = nullptr;
		}
		};
	// ---------------------------------------------------------------
	// First N-R stage: solve the current P/Q operating point from a flat start.
	// On success, angel_res and voltage_res store the true power-flow solution.
	// Do not write failures to the main heatmap because x_true is unavailable.
	// ---------------------------------------------------------------
	auto solve_current_PQ_from_flat_start = [&](int local_max_iter) -> bool {
		nonzero = 0, nnz_flag = 1;

		for (int i = 1; i <= bus_total; i++) {
			nonzeros[i] = 0;
			voltage[i] = np[i].bus.Vm;
			angel[i] = np[i].bus.Va * pi / 180;
		}

		// Compute J_b for the first time and count its nonzero entries.
		stop = 0, nxt_step = pool_size;

		for (; stop < pool_size - 1; stop++) { pool.enqueue([stop, split] {get_J_b(stop * split + 1, (stop + 1) * split, stop, 1); }); }
		pool.enqueue([stop, split] {get_J_b(stop * split + 1, bus_total, stop, 1); });

		WaitForSingleObject(hEvent, INFINITE);
		ResetEvent(hEvent);

		nnz_flag = 0;

		// Index the unknowns: non-slack bus angles followed by PQ-bus voltages.
		int val = 0;
		mp.clear();
		mps.clear();

		for (int i = 1; i <= bus_total; i++) {
			if (types[i] == 3) continue;
			mp[i] = val;
			val++;
			if (types[i] == 1) { mps[i] = val; val++; }
		}

		init_Slover();

		stop = 0;
		nxt_step = pool_size;

		for (; stop < pool_size - 1; stop++) {
			pool.enqueue([stop, split] {
				pre_CSR(stop * split + 1, (stop + 1) * split, stop);
				});
		}
		pool.enqueue([stop, split] {
			pre_CSR(stop * split + 1, bus_total, stop);
			});

		WaitForSingleObject(hEvent, INFINITE);
		ResetEvent(hEvent);

		ap[need_slove] = (_uint_t)nonzero;

		solver_check = NicsLU_Initialize(&solver, &cfg, &stat_p, NULL);
		if (solver_check != 0 || solver == nullptr || cfg == nullptr) {
			release_solver_memory();
			return false;
		}

		cfg[0] = 1.;
		cfg[3] = 4;

		solver_check = solver->Analyze(n_p, ax, ai, ap, MATRIX_REAL_ROW, NULL, NULL, NULL, NULL);
		if (solver_check != 0) {
			release_solver_memory();
			return false;
		}

		solver_check = solver->CreateThreads(pool_size);
		if (solver_check != 0) {
			release_solver_memory();
			return false;
		}

		solver_check = solver->FactorizeMatrix(ax, pool_size);
		if (solver_check != 0) {
			release_solver_memory();
			return false;
		}

		solver_check = solver->Solve(b, x);
		if (solver_check != 0) {
			release_solver_memory();
			return false;
		}

		update_x();

		int step = 1;
		bool pf_converged = false;

		while (step <= local_max_iter) {
			stop = 0;
			nxt_step = pool_size;

			for (; stop < pool_size - 1; stop++) {
				pool.enqueue([stop, split] {
					get_J_b(stop * split + 1, (stop + 1) * split, stop, 2);
					});
			}
			pool.enqueue([stop, split] {
				get_J_b(stop * split + 1, bus_total, stop, 2);
				});

			WaitForSingleObject(hEvent, INFINITE);
			ResetEvent(hEvent);

			if (check_converge() == 0) {
				pf_converged = true;
				break;
			}

			stop = 0;
			nxt_step = pool_size;

			for (; stop < pool_size - 1; stop++) {
				pool.enqueue([stop, split] {
					pre_CSR(stop * split + 1, (stop + 1) * split, stop);
					});
			}
			pool.enqueue([stop, split] {
				pre_CSR(stop * split + 1, bus_total, stop);
				});

			WaitForSingleObject(hEvent, INFINITE);
			ResetEvent(hEvent);

			solver_check = solver->FactorizeMatrix(ax, 0);
			if (solver_check != 0) {
				release_solver_memory();
				return false;
			}

			solver_check = solver->Solve(b, x);
			if (solver_check != 0) {
				release_solver_memory();
				return false;
			}

			update_x();
			step++;
		}

		if (!pf_converged) {
			release_solver_memory();
			return false;
		}

		for (int i = 1; i <= bus_total; i++) {
			angel_res[i] = angel[i];
			voltage_res[i] = voltage[i];
		}

		return true;
		};

	// ---------------------------------------------------------------
	// Main heatmap-test loop.
	// Generate exactly each_cell test samples for every cell.
	// ---------------------------------------------------------------
	auto x_bins = make_bins(X_SCORE_MIN, X_SCORE_MAX, SCORE_STEP);
	auto pq_bins = make_bins(PQ_SCORE_MIN, PQ_SCORE_MAX, SCORE_STEP);

	int test_total = 0;
	double heatmap_max_abs_disP = 0.0;
	double heatmap_max_abs_disQ = 0.0;
	long double heatmap_max_dp_start = 0.0;
	long double heatmap_max_dq_start = 0.0;

	auto snapshot_start_mismatch = [&](
		long double& max_dp_start,
		long double& max_dq_start
		) {
			max_dp_start = 0.0;
			max_dq_start = 0.0;
			for (int i = 1; i <= bus_total; i++) {
				if (types[i] == 3) continue;
				long double dp = std::abs(node[i].p - node[i].right_hp);
				if (dp > max_dp_start) max_dp_start = dp;
				if (types[i] == 1) {
					long double dq = std::abs(node[i].q - node[i].right_hq);
					if (dq > max_dq_start) max_dq_start = dq;
				}
			}
		};

	for (auto xb : x_bins) {
		for (auto pb : pq_bins) {
			int accepted = 0;
			int tries = 0;

			while (accepted < each_cell && tries < max_try_per_cell) {
				tries++;

				// Randomly sample target signed scores within the current cell.
				double target_X_signed = sample_score_in_bin(xb.first, xb.second, generator);
				double target_PQ_signed = sample_score_in_bin(pb.first, pb.second, generator);

				// 1. Construct the current P/Q operating point for the target PQ_signed.
				set_PQ_heatmap(target_PQ_signed, generator);

				// 2. First N-R stage: solve this P/Q operating point from a flat start.
				//    Resample if it does not converge because no reliable x_true is available.
				bool base_pf_ok = solve_current_PQ_from_flat_start(max_iter);

				if (!base_pf_ok) {
					continue;
				}

				// 3. Construct x_init near the true solution for the target X_signed.
				re_init_heatmap(target_X_signed, generator);

				// 4. Verify that the actual X_signed and PQ_signed fall in the current cell.
				update_current_scores();

				if (!score_in_bin(cur_X_signed, xb.first, xb.second) ||
					!score_in_bin(cur_PQ_signed, pb.first, pb.second)) {
					release_solver_memory();
					continue;
				}

				// 5. Second N-R stage: validate from x_init.
				//    Preserve the true solution associated with x_init and record whether
				//    conventional N-R converges from that initial state.
				long double voltage_start_snapshot[need], angel_start_snapshot[need];
				int this_common_sample_id = common_sample_id++;

				int iter_num = 0;
				bool sample_converged = false;
				bool recorded_init = false;

				while (iter_num < max_iter) {
					stop = 0;
					nxt_step = pool_size;

					for (; stop < pool_size - 1; stop++) {
						pool.enqueue([stop, split] {
							get_J_b(stop * split + 1, (stop + 1) * split, stop, 2);
							});
					}
					pool.enqueue([stop, split] {
						get_J_b(stop * split + 1, bus_total, stop, 2);
						});

					WaitForSingleObject(hEvent, INFINITE);
					ResetEvent(hEvent);

					if (iter_num == 0) {
						// Recompute once so meta stores the score associated with x_init.
						update_current_scores();
						for (int ss = 1; ss <= bus_total; ss++) {
							voltage_start_snapshot[ss] = voltage[ss];
							angel_start_snapshot[ss] = angel[ss];
						}

						recorded_init = true;
					}

					if (check_converge() == 0) {
						sample_converged = true;
						break;
					}

					stop = 0;
					nxt_step = pool_size;

					for (; stop < pool_size - 1; stop++) {
						pool.enqueue([stop, split] {
							pre_CSR(stop * split + 1, (stop + 1) * split, stop);
							});
					}
					pool.enqueue([stop, split] {
						pre_CSR(stop * split + 1, bus_total, stop);
						});

					WaitForSingleObject(hEvent, INFINITE);
					ResetEvent(hEvent);

					solver_check = solver->FactorizeMatrix(ax, 0);
					solver_check = solver->Solve(b, x);

					if (solver_check != 0) {
						// Retain linear-solver failures as N-R PF hard-convergence test samples.
						break;
					}

					update_x();
					iter_num++;
				}
				//cout << "iter_num:" << iter_num << endl << endl;
				// 6. Write the common test sample. Preserve the true label and initial state
				//    even when conventional N-R does not converge from this initial value.
				if (recorded_init) {
					for (int ss = 1; ss <= bus_total; ss++) {
						voltage[ss] = voltage_start_snapshot[ss];
						angel[ss] = angel_start_snapshot[ss];
					}
					stop = 0;
					nxt_step = pool_size;
					for (; stop < pool_size - 1; stop++) {
						pool.enqueue([stop, split] {
							get_J_b(stop * split + 1, (stop + 1) * split, stop, 2);
							});
					}
					pool.enqueue([stop, split] {
						get_J_b(stop * split + 1, bus_total, stop, 2);
						});
					WaitForSingleObject(hEvent, INFINITE);
					ResetEvent(hEvent);
					update_current_scores();

					double hm_abs_dA = 0.0, hm_abs_dV = 0.0;
					double hm_abs_disP = 0.0, hm_abs_disQ = 0.0;
					long double hm_dp_start = 0.0, hm_dq_start = 0.0;
					snapshot_current_raw_max(hm_abs_dA, hm_abs_dV, hm_abs_disP, hm_abs_disQ);
					snapshot_start_mismatch(hm_dp_start, hm_dq_start);
					heatmap_max_abs_disP = max(heatmap_max_abs_disP, hm_abs_disP);
					heatmap_max_abs_disQ = max(heatmap_max_abs_disQ, hm_abs_disQ);
					heatmap_max_dp_start = max(heatmap_max_dp_start, hm_dp_start);
					heatmap_max_dq_start = max(heatmap_max_dq_start, hm_dq_start);

					write_common_sample(
						this_common_sample_id,
						"test",
						"heatmap_start",
						sample_converged,
						true,
						iter_num,
						common_meta_out,
						common_bus_state_out,
						common_jacobian_out,
						common_branch_flow_out,
						xb.first,
						xb.second,
						pb.first,
						pb.second
					);

					if (!sample_converged) {
						printf("N-R PF hard-convergence sample generated\n");
					}

					accepted++;
					test_total++;
				}

				release_solver_memory();
			}

			if (accepted < each_cell) {
				cout << "Warning: cell X=(" << xb.first << "," << xb.second
					<< "), PQ=(" << pb.first << "," << pb.second
					<< ") only accepted " << accepted << "/" << each_cell
					<< " after " << tries << " tries." << endl;
			}
			else {
				cout << "Cell done: X=(" << xb.first << "," << xb.second
					<< "), PQ=(" << pb.first << "," << pb.second
					<< "), accepted = " << accepted << endl;
			}
		}
	}

	cout << "Heatmap test generation done. test_total = " << test_total << endl;
	// ================================================================
	// Stage 3: Generate N-R PF numerical-failure test data.
	// Search close to the training/OOD region first, primarily enlarging
	// the initial-state perturbation X. P/Q stress is expanded only slowly.
	// Only genuine N-R numerical failures are retained: linear-solver failure
	// or clear numerical divergence. Slow-but-convergent cases are rejected.
	// ================================================================
	cout << "****************************************************" << endl;
	cout << "Begin N-R PF numerical-failure test generation..." << endl;
	cout << "****************************************************" << endl;

	const int ill_target_total = 100;
	const int ill_base_pf_max_iter = 30;
	const int ill_verify_max_iter = 50;

	// Conservative divergence detection. These thresholds are deliberately
	// much looser than the PF convergence tolerance so that temporary Newton
	// overshoots are not mislabeled as divergence.
	const double ill_div_abs_residual = 1.0e5;
	const double ill_div_rel_residual = 1.0e4;
	const double ill_growth_ratio = 1.20;
	const double ill_growth_over_initial = 10.0;
	const int ill_growth_patience = 6;

	enum class NRStatus {
		CONVERGED,
		LINEAR_FAIL,
		DIVERGED,
		MAX_ITER,
		SETUP_FAIL
	};

	struct NRVerifyResult {
		NRStatus status = NRStatus::MAX_ITER;
		int iter = 0;
		double r0 = std::numeric_limits<double>::infinity();
		double best_r = std::numeric_limits<double>::infinity();
		double final_r = std::numeric_limits<double>::infinity();
		double max_r = 0.0;
	};

	struct IllSearchBand {
		double x_min;
		double x_max;
		double pq_min;
		double pq_max;
		long long max_try;
	};

	// Ordered from near-training/mild-OOD to stronger initial-state stress.
	// P/Q remains deliberately much closer to the training/OOD region than X.
	const std::vector<IllSearchBand> ill_bands = {
		{0.80, 1.20, 0.00, 1.00,  5000},
		{1.20, 1.50, 0.00, 1.00,  8000},
		{1.50, 1.80, 0.00, 1.10, 12000},
		{1.80, 2.10, 0.00, 1.20, 18000},
		{2.10, 2.40, 0.20, 1.30, 25000},
		{2.40, 2.80, 0.40, 1.40, 35000},
		{2.80, 3.20, 0.60, 1.50, 45000}
	};

	auto nr_status_name = [&](NRStatus s) -> const char* {
		switch (s) {
		case NRStatus::CONVERGED:   return "CONVERGED";
		case NRStatus::LINEAR_FAIL: return "LINEAR_FAIL";
		case NRStatus::DIVERGED:    return "DIVERGED";
		case NRStatus::MAX_ITER:    return "MAX_ITER";
		case NRStatus::SETUP_FAIL:  return "SETUP_FAIL";
		default:                    return "UNKNOWN";
		}
		};

	// Exact current infinity norm of the active/reactive PF mismatch.
	auto get_current_Rinf = [&]() -> double {
		double r_inf = 0.0;
		for (int i = 1; i <= bus_total; i++) {
			if (types[i] == 3) continue;

			double dp = std::abs(static_cast<double>(node[i].p - node[i].right_hp));
			r_inf = max(r_inf, dp);

			if (types[i] == 1) {
				double dq = std::abs(static_cast<double>(node[i].q - node[i].right_hq));
				r_inf = max(r_inf, dq);
			}
		}
		return r_inf;
		};

	auto nr_state_is_finite = [&]() -> bool {
		for (int i = 1; i <= bus_total; i++) {
			if (!std::isfinite(static_cast<double>(voltage[i])) ||
				!std::isfinite(static_cast<double>(angel[i]))) {
				return false;
			}
		}
		return true;
		};

	// Ill-specific initial-state perturbation. The dominant perturbation is
	// placed on a voltage angle, while voltage-magnitude perturbations remain
	// close to the observed training scale. This stresses the N-R attraction
	// basin without creating unnecessarily extreme voltage magnitudes.
	auto re_init_ill = [&](double target_X_signed) {
		double abs_score = std::abs(target_X_signed);
		double sign = target_X_signed >= 0.0 ? 1.0 : -1.0;

		std::vector<int> angle_candidates;
		for (int i = 1; i <= bus_total; i++) {
			if (types[i] != 3) angle_candidates.push_back(i);
		}

		double small_angle_score = 0.30 * abs_score;
		double small_voltage_score = min(0.80, 0.30 * abs_score);

		for (int i = 1; i <= bus_total; i++) {
			if (types[i] == 3) continue;

			angel[i] = angel_res[i] +
				rand_uniform(-small_angle_score, small_angle_score, generator) * train_max_dA;

			if (types[i] == 1) {
				voltage[i] = voltage_res[i] +
					rand_uniform(-small_voltage_score, small_voltage_score, generator) * train_max_dV;
			}
			else if (types[i] == 2) {
				voltage[i] = voltage_res[i];
			}
		}

		if (!angle_candidates.empty()) {
			int idx = static_cast<int>(rand_uniform(
				0.0, static_cast<double>(angle_candidates.size()), generator));
			if (idx >= static_cast<int>(angle_candidates.size())) {
				idx = static_cast<int>(angle_candidates.size()) - 1;
			}
			int bus = angle_candidates[idx];
			angel[bus] = angel_res[bus] + sign * abs_score * train_max_dA;
		}
		};

	// ---------------------------------------------------------------
	// Rebuild the linear system from the current voltage, angle, P, and Q.
	// This preserves x_init and only rebuilds J, b, and CSR.
	// ---------------------------------------------------------------
	auto build_current_start_linear_system = [&]() -> bool {
		release_solver_memory();

		nonzero = 0;
		nnz_flag = 1;

		for (int i = 1; i <= bus_total; i++) {
			nonzeros[i] = 0;
		}

		stop = 0;
		nxt_step = pool_size;

		for (; stop < pool_size - 1; stop++) {
			pool.enqueue([stop, split] {
				get_J_b(stop * split + 1, (stop + 1) * split, stop, 1);
				});
		}

		pool.enqueue([stop, split] {
			get_J_b(stop * split + 1, bus_total, stop, 1);
			});

		WaitForSingleObject(hEvent, INFINITE);
		ResetEvent(hEvent);
		nnz_flag = 0;

		int val = 0;
		mp.clear();
		mps.clear();

		for (int i = 1; i <= bus_total; i++) {
			if (types[i] == 3) continue;
			mp[i] = val++;
			if (types[i] == 1) mps[i] = val++;
		}

		init_Slover();

		stop = 0;
		nxt_step = pool_size;

		for (; stop < pool_size - 1; stop++) {
			pool.enqueue([stop, split] {
				pre_CSR(stop * split + 1, (stop + 1) * split, stop);
				});
		}
		pool.enqueue([stop, split] {
			pre_CSR(stop * split + 1, bus_total, stop);
			});

		WaitForSingleObject(hEvent, INFINITE);
		ResetEvent(hEvent);

		ap[need_slove] = (_uint_t)nonzero;
		return true;
		};

	// ---------------------------------------------------------------
	// Run N-R from the current x_init. The result stores the best residual
	// attained by N-R, which is printed to the terminal for later Fig. 3 use.
	// ---------------------------------------------------------------
	auto run_nr_verify_from_current_start = [&](int verify_max_iter) -> NRVerifyResult {
		NRVerifyResult result;

		solver_check = NicsLU_Initialize(&solver, &cfg, &stat_p, NULL);
		if (solver_check != 0 || solver == nullptr || cfg == nullptr) {
			result.status = NRStatus::SETUP_FAIL;
			return result;
		}

		cfg[0] = 1.;
		cfg[3] = 4;

		solver_check = solver->Analyze(
			n_p, ax, ai, ap, MATRIX_REAL_ROW,
			NULL, NULL, NULL, NULL);
		if (solver_check != 0) {
			result.status = NRStatus::LINEAR_FAIL;
			return result;
		}

		solver_check = solver->CreateThreads(pool_size);
		if (solver_check != 0) {
			result.status = NRStatus::SETUP_FAIL;
			return result;
		}

		double current_r = get_current_Rinf();
		result.r0 = current_r;
		result.best_r = current_r;
		result.final_r = current_r;
		result.max_r = current_r;

		if (!std::isfinite(current_r) || !nr_state_is_finite()) {
			result.status = NRStatus::DIVERGED;
			return result;
		}

		double prev_r = current_r;
		int growth_count = 0;

		while (result.iter < verify_max_iter) {
			if (current_r <= check_to_end) {
				result.status = NRStatus::CONVERGED;
				return result;
			}

			solver_check = solver->FactorizeMatrix(
				ax, result.iter == 0 ? pool_size : 0);
			if (solver_check != 0) {
				result.status = NRStatus::LINEAR_FAIL;
				return result;
			}

			solver_check = solver->Solve(b, x);
			if (solver_check != 0) {
				result.status = NRStatus::LINEAR_FAIL;
				return result;
			}

			update_x();
			result.iter++;

			if (!nr_state_is_finite()) {
				result.status = NRStatus::DIVERGED;
				return result;
			}

			// Recompute current mismatch and Jacobian values.
			stop = 0;
			nxt_step = pool_size;

			for (; stop < pool_size - 1; stop++) {
				pool.enqueue([stop, split] {
					get_J_b(stop * split + 1, (stop + 1) * split, stop, 2);
					});
			}
			pool.enqueue([stop, split] {
				get_J_b(stop * split + 1, bus_total, stop, 2);
				});

			WaitForSingleObject(hEvent, INFINITE);
			ResetEvent(hEvent);

			current_r = get_current_Rinf();
			result.final_r = current_r;
			result.best_r = min(result.best_r, current_r);
			result.max_r = max(result.max_r, current_r);

			if (!std::isfinite(current_r)) {
				result.status = NRStatus::DIVERGED;
				return result;
			}

			if (current_r > prev_r * ill_growth_ratio) growth_count++;
			else growth_count = 0;

			double safe_r0 = max(result.r0, check_to_end);
			bool residual_explosion =
				current_r > ill_div_abs_residual ||
				current_r > ill_div_rel_residual * safe_r0;

			bool sustained_growth =
				growth_count >= ill_growth_patience &&
				current_r > ill_growth_over_initial * safe_r0 &&
				current_r > 1.0e-2;

			if (residual_explosion || sustained_growth) {
				result.status = NRStatus::DIVERGED;
				return result;
			}

			if (current_r <= check_to_end) {
				result.status = NRStatus::CONVERGED;
				return result;
			}

			prev_r = current_r;

			if (result.iter >= verify_max_iter) break;

			// Rebuild CSR from the refreshed Jacobian before the next step.
			stop = 0;
			nxt_step = pool_size;

			for (; stop < pool_size - 1; stop++) {
				pool.enqueue([stop, split] {
					pre_CSR(stop * split + 1, (stop + 1) * split, stop);
					});
			}
			pool.enqueue([stop, split] {
				pre_CSR(stop * split + 1, bus_total, stop);
				});

			WaitForSingleObject(hEvent, INFINITE);
			ResetEvent(hEvent);
		}

		// A maximum-iteration case is accepted as divergence only when its
		// trajectory has clearly blown up. Purely slow/stagnating trajectories
		// remain MAX_ITER and are NOT retained in the failure test set.
		double safe_r0 = max(result.r0, check_to_end);
		if (result.final_r > max(10.0 * safe_r0, 100.0 * result.best_r) &&
			result.final_r > 1.0e-2) {
			result.status = NRStatus::DIVERGED;
		}
		else {
			result.status = NRStatus::MAX_ITER;
		}

		return result;
		};

	int ill_total = 0;
	long long ill_try = 0;
	int ill_linear_fail = 0;
	int ill_diverged = 0;

	// Run one adaptive search band. The returned samples keep exactly the
	// original output schema and source="ill-conditioned" for Python compatibility.
	auto search_ill_band = [&](const IllSearchBand& band, int band_id, const char* band_name) {
		long long band_try = 0;
		int accepted_before = ill_total;

		cout << "[NR search] " << band_name << " " << band_id
			<< " X=[" << band.x_min << "," << band.x_max << "]"
			<< " PQ=[" << band.pq_min << "," << band.pq_max << "]"
			<< " max_try=" << band.max_try << endl;

		while (ill_total < ill_target_total && band_try < band.max_try) {
			band_try++;
			ill_try++;

			if (band_try % 5000 == 0) {
				cout << "[NR search progress] band_try=" << band_try
					<< " total_try=" << ill_try
					<< " accepted=" << ill_total << "/" << ill_target_total << endl;
			}

			double x_abs_score = rand_uniform(band.x_min, band.x_max, generator);
			double pq_abs_score = rand_uniform(band.pq_min, band.pq_max, generator);

			double x_sign = rand_uniform(0.0, 1.0, generator) < 0.5 ? -1.0 : 1.0;
			double pq_sign = rand_uniform(0.0, 1.0, generator) < 0.5 ? -1.0 : 1.0;

			double target_X_signed = x_sign * x_abs_score;
			double target_PQ_signed = pq_sign * pq_abs_score;

			// 1. Construct a P/Q operating point and require a valid PF solution.
			set_PQ_heatmap(target_PQ_signed, generator);
			bool base_pf_ok = solve_current_PQ_from_flat_start(ill_base_pf_max_iter);
			if (!base_pf_ok) {
				release_solver_memory();
				continue;
			}

			// 2. Stress mainly the initial voltage state, especially voltage angles.
			re_init_ill(target_X_signed);
			update_current_scores();

			bool in_current_band =
				std::abs(cur_X_signed) >= band.x_min - 1.0e-9 &&
				std::abs(cur_X_signed) <= band.x_max + 1.0e-9 &&
				std::abs(cur_PQ_signed) >= band.pq_min - 1.0e-9 &&
				std::abs(cur_PQ_signed) <= band.pq_max + 1.0e-9;

			if (!in_current_band) {
				release_solver_memory();
				continue;
			}

			// 3. Preserve x_init because N-R verification modifies it.
			long double voltage_start_snapshot[need];
			long double angel_start_snapshot[need];
			for (int ss = 1; ss <= bus_total; ss++) {
				voltage_start_snapshot[ss] = voltage[ss];
				angel_start_snapshot[ss] = angel[ss];
			}

			if (!build_current_start_linear_system()) {
				release_solver_memory();
				continue;
			}

			NRVerifyResult nr_result =
				run_nr_verify_from_current_start(ill_verify_max_iter);

			bool is_nr_failure =
				nr_result.status == NRStatus::LINEAR_FAIL ||
				nr_result.status == NRStatus::DIVERGED;

			if (!is_nr_failure) {
				release_solver_memory();
				continue;
			}

			// 4. Restore the original x_init before writing exactly the same output format.
			for (int ss = 1; ss <= bus_total; ss++) {
				voltage[ss] = voltage_start_snapshot[ss];
				angel[ss] = angel_start_snapshot[ss];
			}

			if (!build_current_start_linear_system()) {
				release_solver_memory();
				continue;
			}

			update_current_scores();
			int this_common_sample_id = common_sample_id++;

			write_common_sample(
				this_common_sample_id,
				"test",
				"ill-conditioned", // Keep legacy metadata value for loader compatibility.
				false,
				true,
				nr_result.iter,
				common_meta_out,
				common_bus_state_out,
				common_jacobian_out,
				common_branch_flow_out,
				-band.x_max,
				band.x_max,
				-band.pq_max,
				band.pq_max
			);

			ill_total++;
			if (nr_result.status == NRStatus::LINEAR_FAIL) ill_linear_fail++;
			if (nr_result.status == NRStatus::DIVERGED) ill_diverged++;

			cout << "[NR-FAIL " << ill_total << "/" << ill_target_total << "]"
				<< " sample_id=" << this_common_sample_id
				<< " type=" << nr_status_name(nr_result.status)
				<< " iter=" << nr_result.iter
				<< " R0=" << nr_result.r0
				<< " Rbest=" << nr_result.best_r
				<< " Rfinal=" << nr_result.final_r
				<< " Rmax=" << nr_result.max_r
				<< " X=" << cur_X_signed
				<< " PQ=" << cur_PQ_signed
				<< " band=" << band_name << "-" << band_id
				<< endl;

			release_solver_memory();
		}

		cout << "[NR search done] " << band_name << " " << band_id
			<< " tries=" << band_try
			<< " newly_accepted=" << (ill_total - accepted_before)
			<< " total=" << ill_total << "/" << ill_target_total << endl;
		};

	// Pass 1: deterministic near-to-far bands.
	for (size_t band_id = 0;
		band_id < ill_bands.size() && ill_total < ill_target_total;
		++band_id) {
		search_ill_band(ill_bands[band_id], static_cast<int>(band_id), "preset");
	}

	// Pass 2: persistent fallback. This stage does not stop below 100 samples.
	// It continues to enlarge X first. P/Q expands only after X becomes large,
	// and is capped near the tested OOD scale to avoid generating trivially
	// pathological operating points solely by extreme injection stress.
	int fallback_round = 0;
	double fb_x_min = 3.00;
	double fb_x_max = 3.50;
	double fb_pq_min = 0.50;
	double fb_pq_max = 1.50;
	const double fb_x_cap = 8.00;
	const double fb_pq_cap = 2.00;
	const long long fb_try_per_round = 50000;

	while (ill_total < ill_target_total) {
		IllSearchBand band{
			fb_x_min,
			fb_x_max,
			fb_pq_min,
			fb_pq_max,
			fb_try_per_round
		};

		search_ill_band(band, fallback_round, "fallback");
		if (ill_total >= ill_target_total) break;

		fallback_round++;

		// Expand the initial-state range first.
		if (fb_x_max < fb_x_cap - 1.0e-12) {
			fb_x_min = max(0.0, fb_x_max - 0.20);
			fb_x_max = min(fb_x_cap, fb_x_max + 0.50);
		}
		else {
			// Only after reaching the X cap, slowly enlarge P/Q stress.
			fb_pq_max = min(fb_pq_cap, fb_pq_max + 0.10);
			fb_pq_min = max(0.0, fb_pq_min - 0.05);
		}

		cout << "[NR fallback expand] next X=[" << fb_x_min << "," << fb_x_max
			<< "] PQ=[" << fb_pq_min << "," << fb_pq_max << "]"
			<< " accepted=" << ill_total << "/" << ill_target_total << endl;

		// Once both caps are reached the generator keeps sampling this broadest
		// admissible band until 100 genuine failures are collected. This avoids
		// silently terminating with fewer than the requested 100 cases.
	}

	cout << "N-R PF numerical-failure generation done." << endl;
	cout << "N-R PF numerical-failure total search attempts = " << ill_try << endl;
	cout << "N-R PF numerical-failure test case number = " << ill_total << endl;
	cout << "  LINEAR_FAIL = " << ill_linear_fail << endl;
	cout << "  DIVERGED    = " << ill_diverged << endl;
	return 0;
}