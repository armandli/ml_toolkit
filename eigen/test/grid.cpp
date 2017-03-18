#include <cstring>
#include <ctime>
#include <iostream>
#include <random>

#include <Eigen/Dense>

#include <ml_util.h>
#include <ml_ffn.h>

using Eigen::MatrixXd;

#define SIZE 4
#define PMASK 0x1
#define WMASK 0x2
#define HMASK 0x4
#define GMASK 0x8

enum Move: int {
  UP = 0,
  DN,
  LF,
  RG,
};

int MArray[][2] = {{-1, 0},{1, 0}, {0, -1}, {0, 1}};

struct Game {
  int board[SIZE][SIZE];
  Game(){ memset(board, 0, sizeof(int) * SIZE * SIZE); }
};

/* simple basic game */
Game create_game1(){
  Game ret;
  ret.board[0][1] ^= PMASK;
  ret.board[2][2] ^= WMASK;
  ret.board[1][1] ^= HMASK;
  ret.board[3][3] ^= GMASK;
  return ret;
}

void make_move(Game& game, Move move){
  int x, y, a, b;
  for (int i = 0; i < SIZE; ++i)
    for (int j = 0; j < SIZE; ++j)
      if (game.board[i][j] & PMASK){
        a = i; b = j;
        goto m;
      }
m:switch (move){
    case UP: x = a + MArray[UP][0]; y = b + MArray[UP][1]; break;
    case DN: x = a + MArray[DN][0]; y = b + MArray[DN][1]; break;
    case LF: x = a + MArray[LF][0]; y = b + MArray[LF][1]; break;
    case RG: x = a + MArray[RG][0]; y = b + MArray[RG][1]; break;
    default: assert(false);
  }
  if (x < 0 || y < 0 || x >= 4 || y >= 4) return;
  if (game.board[x][y] & WMASK) return;
  
  game.board[x][y] |= PMASK;
  game.board[a][b] ^= PMASK;
}

double get_reward(Game& game){
  for (int i = 0; i < SIZE; ++i)
    for (int j = 0; j < SIZE; ++j)
      if (game.board[i][j] & PMASK){
        if (game.board[i][j] & HMASK) return -10.;
        if (game.board[i][j] & GMASK) return 10.;
        goto m;
      }
m:return -1.;
}

ostream& operator<<(ostream& out, const Game& game){
  for (int i =0; i < SIZE; ++i){
    out << "|";
    for (int j = 0; j < SIZE; ++j){
      if (game.board[i][j] & WMASK)      out << "W";
      else if (game.board[i][j] & HMASK) out << "H";
      else if (game.board[i][j] & GMASK) out << "G";
      else                               out << " ";
      if (game.board[i][j] & PMASK) out << "+";
      else                          out << " ";
    }
    out << "|" << endl;
  }
  return out;
}

MatrixXd to_mtx(const Game& game){
  MatrixXd ret(1, SIZE * SIZE * 4);
  for (int i = 0; i < SIZE; ++i)
    for (int j = 0; j < SIZE; ++j){
      ret(0, i * SIZE + j) = game.board[i][j] & PMASK ? 1. : 0.;
      ret(0, i * SIZE + j + SIZE * SIZE) = game.board[i][j] & WMASK ? 1. : 0.;
      ret(0, i * SIZE + j + SIZE * SIZE * 2) = game.board[i][j] & HMASK ? 1. : 0.;
      ret(0, i * SIZE + j + SIZE * SIZE * 3) = game.board[i][j] & GMASK ? 1. : 0.;
    }
  return ret;
}

using NN = FFN<MSE, ReluFun, L2Reg, AdagradUpdate, false>;

void train(NN& model){
  uniform_real_distribution<double> random_step(0., 1.);
  uniform_int_distribution<int> random_choice(0, 3);

  int epochs = 1000;
  double gamma = 0.9;
  double epsilon = 1.;

  for (int k = 0; k < epochs; ++k){
    Game game = create_game1();
    cout << "game " << k << endl;
    int step_count = 0;
    double reward = -1.;
    while (reward == -1.){
      step_count++;

      MatrixXd X = to_mtx(game);
      MatrixXd O = model.predict(X);

      int action = 0;
      if (random_step(get_default_random_engine()) < epsilon){
        action = random_choice(get_default_random_engine());
      } else {
        MatrixXd::Index idx;
        O.row(0).maxCoeff(&idx);
        action = idx;
      }
      make_move(game, (Move)action);
      reward = get_reward(game);
      MatrixXd Xn = to_mtx(game);
      MatrixXd On = model.predict(Xn);
      double maxQ = On.maxCoeff();
      MatrixXd Y = O;
      double update;
      if (reward == -1.) update = reward + gamma * maxQ;
      else               update = reward;
      Y(0, action) = update;
      model.train(X, Y);
    }
    if (epsilon > .1) epsilon -= 1 / epochs;

    cout << "steps: " << step_count << " reward: " << reward << endl;
  }
}


void play(NN& model){
  vector<Move> moves;
  Game game = create_game1();
  cout << game << endl;
  double reward = get_reward(game);
  for (int i = 0; i < 40 && reward != 10. && reward != -10.; ++i){
    MatrixXd X = to_mtx(game);
    MatrixXd O = model.predict(X);
    MatrixXd::Index idx; O.row(0).maxCoeff(&idx);
    Move action = (Move)idx;
    make_move(game, action);
    reward = get_reward(game);
    moves.push_back(action);
    cout << game << endl;
  }
}

int main(){
  vector<int> dims;
  dims.push_back(SIZE * SIZE * 4);
  dims.push_back(164);
  dims.push_back(150);
  dims.push_back(4);
  NN model(dims, 10, 0.01, 0.0001, false);
  train(model);
  play(model);

//  Game game = create_game1();
//  cout << game;
//  make_move(game, DN);
//  cout << game;
}
