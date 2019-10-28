data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> T;
  int<lower=1,upper=T> team[N];
  int<lower=1,upper=T> team_opp[N];
  matrix[N,K] x;
  vector[N] logit_y;
  int<lower=0,upper=1> likelihood;
}
transformed data {
  matrix[N,K] x_std;
  for (k in 1:K){
    x_std[,k] = (x[,k] - mean(x[,k])) / (2 * sd(x[,k]));
  }
}
parameters {
  real a;
  vector[T] a_team;
  vector[T] a_team_opp;
  vector[K] b;
  real<lower=0> sigma;
  real<lower=0> sigma_team;
  real<lower=0> sigma_team_opp;
}
model {
  a_team ~ normal(0,sigma_team);
  a_team_opp ~ normal(0,sigma_team_opp);
  b ~ normal(0,1);
  a ~ normal(-4, 2);
  sigma ~ normal(0, 1);
  sigma_team ~ normal(0, 1);
  sigma_team_opp ~ normal(0, 1);
  if (likelihood == 1){
    vector[N] eta = a + a_team[team] + a_team_opp[team_opp] + x_std * b;
    logit_y ~ normal(eta,sigma);
  }
}
generated quantities {
  vector[N] logit_y_rep;
  for (n in 1:N){
    real eta_rep = a + a_team[team[n]] + a_team_opp[team_opp[n]] + x_std[n] * b;
    logit_y_rep[n] = normal_rng(eta_rep, sigma);
  }
}
