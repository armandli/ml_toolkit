#include <random>

#include <ml_util.h>
#include <ml_ffn.h>
#include <ml_rl.h>

using namespace std;
using Eigen::MatrixXd;

class GridGame : public GameModelIF {
  static const size_t GridSize = 4;
  static const size_t PMASK = 0x1;
  static const size_t WMASK = 0x2;
  static const size_t HMASK = 0x3;
  static const size_t GMASK = 0x4;
  static const int MArray[4][2];

  int board1[GridSize][GridSize];
  int board2[GridSize][GridSize];
  bool is_board1;
  size_t steps;
public:
  GridGame(): is_board1(true), steps(0) {}

  virtual void init() final {
    memset(board1, 0, GridSize * GridSize * sizeof(int));
    memset(board2, 0, GridSize * GridSize * sizeof(int));

    board1[2][2] ^= WMASK;
    board1[2][2] ^= HMASK;
    board1[2][2] ^= GMASK;

    bool not_found = true;
    uniform_int_distribution<int> dist(0, GridSize - 1);

    while (not_found){
      int a = dist(get_default_random_engine()), b = dist(get_default_random_engine());
      if (board1[a][b] == 0){
        board1[a][b] ^= PMASK;
        not_found = false;
      }
    }

    steps = 0;
  }

  virtual int get_input_dimension() final {
    return GridSize * GridSize * 4;
  }

  virtual int get_output_dimension() final {
    return 4;
  }

  virtual MatrixXd to_input() final {
    MatrixXd ret(1, GridSize * GridSize * 4);
    
    int (&board)[GridSize][GridSize] = is_board1 ? board1 : board2;
    for (size_t i = 0; i < GridSize; ++i)
      for (size_t j = 0; j < GridSize; ++j){
        ret(0, i * GridSize + j) =                           board[i][j] & PMASK ? 1. : 0.;
        ret(0, i * GridSize + j + GridSize * GridSize) =     board[i][j] & WMASK ? 1. : 0.;
        ret(0, i * GridSize + j + GridSize * GridSize * 2) = board[i][j] & HMASK ? 1. : 0.;
        ret(0, i * GridSize + j + GridSize * GridSize * 3) = board[i][j] & GMASK ? 1. : 0.;
      }
    return ret;
  }

  virtual void make_move(int move) final {
    assert(move >= 0 && move < 4);

    if (is_board1) memcpy(board2, board1, GridSize * GridSize * sizeof(int));
    else           memcpy(board1, board2, GridSize * GridSize * sizeof(int));
    is_board1 ^= true;

    int (&board)[GridSize][GridSize] = is_board1 ? board1 : board2;

    int x, y, a, b;
    for (size_t i = 0; i < GridSize; ++i)
      for (size_t j = 0; j < GridSize; ++j)
        if (board[i][j] & PMASK){
          a = i; b = j;
          goto m;
        }
m:  switch (move){
      case 0: x = a + MArray[0][0]; y = b + MArray[0][1]; break;
      case 1: x = a + MArray[1][0]; y = b + MArray[1][1]; break;
      case 2: x = a + MArray[2][0]; y = b + MArray[2][1]; break;
      case 3: x = a + MArray[3][0]; y = b + MArray[3][1]; break;
      default: assert(false); break;
    }
    if (x < 0 || y < 0 || x >= GridSize || y >= GridSize) return;
    if (board[x][y] & WMASK) return;

    board[x][y] |= PMASK;
    board[a][b] ^= PMASK;

    steps++;
  }

  virtual double get_reward() final {
    int (&board)[GridSize][GridSize] = is_board1 ? board1 : board2;
    size_t gx = 0, gy = 0;
    size_t px = 0, py = 0;
    for (size_t i = 0; i < GridSize; ++i)
      for (size_t j = 0; j < GridSize; ++j){
        if (board[i][j] & GMASK){
          gx = i; gy = j;
        }
        if (board[i][j] & PMASK){
          px = i; py = j;
          if (board[i][j] & HMASK) return -10.;
          if (board[i][j] & GMASK) return 10.;
        }
      }

    int (&antiboard)[GridSize][GridSize] = is_board1 ? board2 : board1;
    bool board_equal = true;
    for (size_t i = 0; i < GridSize; ++i)
      for (size_t j = 0; j < GridSize; ++j)
        if (antiboard[i][j] != board[i][j]){
          board_equal = false;
          break;
        }
    if (board_equal) return -9.;

    return (abs(px - gx) + abs(py - gy)) * -1.;
  }

  virtual bool is_terminal_state() final {
    int (&board)[GridSize][GridSize] = is_board1 ? board1 : board2;
    for (size_t i = 0; i < GridSize; ++i)
      for (size_t j = 0; j < GridSize; ++j)
        if (board[i][j] & PMASK){
          if (board[i][j] & HMASK || board[i][j] & GMASK) return true;
          break;
        }
    return false;
  }

  virtual ostream& report_stat(ostream& out) final {
    out << "steps: " << steps << endl;
    return out;
  }
};
const size_t GridGame::GridSize;
const size_t GridGame::PMASK;
const size_t GridGame::WMASK;
const size_t GridGame::HMASK;
const size_t GridGame::GMASK;
const int GridGame::MArray[][2] = {{-1, 0},{1, 0}, {0, -1}, {0, 1}};

int main(){
  GridGame game;
  int input_size = game.get_input_dimension();
  int output_size = game.get_output_dimension();
  vector<int> model_dims = {input_size, 164, 150, output_size};
  FFN<MSE, ReluFun, L2Reg, AdamUpdate, false> model(model_dims, 10, 0.03, 0.0001, false);
  RFL<decltype(model)> rflearner(game, model);

  rflearner.train(500);
}
