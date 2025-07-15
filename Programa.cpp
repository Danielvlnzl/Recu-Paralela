#include <bits/stdc++.h>
#include <omp.h>
using namespace std;
struct Edge { int to; long long w; };
constexpr long long INF = 4e18;

vector<vector<long long>> parse_matrix(const string &s) {
  
    vector<vector<long long>> mat; vector<long long> row; string num; bool on = false;
  
    for (char c: s) {
      
        if (isdigit(c)) { num.push_back(c); on = true; }
          
        else { if (on) { row.push_back(stoll(num)); num.clear(); on = false; } if (c==']' && !row.empty()) { mat.push_back(row); row.clear(); } }
    }
  
    return mat;
}

vector<vector<Edge>> to_adj(const vector<vector<long long>> &mat) {
  
    int n=mat.size(); vector<vector<Edge>> g(n);
  
    for(int u=0;u<n;++u) for(int v=0;v<n;++v) if(mat[u][v] && u!=v) g[u].push_back({v,mat[u][v]});
  
    return g;
}

vector<long long> dijkstra_parallel_multiq(const vector<vector<Edge>>& g,int src){
  
    int n=g.size(); vector<long long> dist(n,INF); dist[src]=0;
  
    using P=pair<long long,int>;
  
    int T=omp_get_max_threads();
  
    vector< priority_queue<P,vector<P>,greater<P>> > queues(T);
  
    queues[0].push({0,src});

  
    vector<omp_lock_t> vlocks(n), qlocks(T);
  
    for(auto &lk:vlocks) omp_init_lock(&lk);
  
    for(auto &lk:qlocks) omp_init_lock(&lk);

    #pragma omp parallel default(none) shared(queues,qlocks,dist,g,vlocks,T)
  
    {
        int tid=omp_get_thread_num();
      
        while(true){
          
            P cur; bool have=false;

            omp_set_lock(&qlocks[tid]);
          
            if(!queues[tid].empty()){ cur=queues[tid].top(); queues[tid].pop(); have=true; }
          
            omp_unset_lock(&qlocks[tid]);

            if(!have){
              
                for(int k=1;k<=T && !have;++k){
                  
                    int victim=(tid+k)%T;
                  
                    omp_set_lock(&qlocks[victim]);
                  
                    if(!queues[victim].empty()){
                      
                        cur=queues[victim].top(); queues[victim].pop(); have=true;
                    }
                  
                    omp_unset_lock(&qlocks[victim]);
                }
              
                if(!have){
                  
                    bool empty=true;
                  
                    for(int i=0;i<T;++i){ omp_set_lock(&qlocks[i]); empty&=queues[i].empty(); omp_unset_lock(&qlocks[i]); if(!empty)break; }
                  
                    if(empty) break; else continue;
                }
            }

            auto [du,u]=cur; if(du!=dist[u]) continue;
          
            for(const auto &e:g[u]){
              
                int v=e.to; long long alt=du+e.w; if(alt<dist[v]){
                  
                    omp_set_lock(&vlocks[v]); if(alt<dist[v]){ dist[v]=alt; omp_set_lock(&qlocks[tid]); queues[tid].push({alt,v}); omp_unset_lock(&qlocks[tid]); } omp_unset_lock(&vlocks[v]);
                }
            }
        }
    }

    for(auto &lk:vlocks) omp_destroy_lock(&lk);
  
    for(auto &lk:qlocks) omp_destroy_lock(&lk);
  
    return dist;
}

int main(int argc,char*argv[]){
  
    if(argc!=4){ cerr<<"Uso: "<<argv[0]<<" \"[matriz]\" origen archivo_salida\n"; return 1; }
  
    auto mat=parse_matrix(argv[1]); int src=stoi(argv[2]); string out=argv[3];
  
    if(src<0||src>=mat.size()){ cerr<<"Origen fuera de rango\n"; return 1; }
  
    auto g=to_adj(mat);
  
    double t0=omp_get_wtime(); auto dist=dijkstra_parallel_multiq(g,src); double t1=omp_get_wtime();
  
    ofstream f(out); if(!f){ cerr<<"No puedo abrir "<<out<<"\n"; return 1; }
  
    f<<"Vértice\tDistancia\n"; for(int i=0;i<dist.size();++i) f<<i<<"\t"<<(dist[i]==INF?-1:dist[i])<<"\n"; f<<"\nTiempo: "<<fixed<<setprecision(6)<<t1-t0<<" s\n";
  
    cout<<"Resultados guardados en "<<out<<"\n"; return 0;
}
