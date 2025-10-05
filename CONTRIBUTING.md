# 貢献ガイドライン (Contribution Guidelines)

## 目次 (Table of Contents)

1. [貢献の種類](#貢献の種類)
2. [開発プロセス](#開発プロセス)
3. [コーディング標準](#コーディング標準)
4. [テスト](#テスト)
5. [ドキュメント](#ドキュメント)
6. [プルリクエスト](#プルリクエスト)
7. [レビュープロセス](#レビュープロセス)
8. [コミュニティ](#コミュニティ)

## 貢献の種類 (Types of Contributions)

### コード貢献 (Code Contributions)
- 新機能の実装
- バグ修正
- パフォーマンス改善
- リファクタリング
- セキュリティ強化

### ドキュメント貢献 (Documentation Contributions)
- READMEの改善
- APIドキュメントの更新
- チュートリアル作成
- 翻訳の追加
- コメントの改善

### コミュニティ貢献 (Community Contributions)
- バグ報告
- 機能リクエスト
- 質問と回答
- サポートの提供
- フィードバック

## 開発プロセス (Development Process)

### 1. 環境のセットアップ
```bash
# リポジトリのフォーク
git clone <フォークしたリポジトリのURL>
cd <クローンしたディレクトリ>

# 開発環境のセットアップ
python -m venv dev-env
source dev-env/bin/activate  # Linux/macOS
# dev-env\Scripts\activate  # Windows

# 依存関係のインストール
pip install -r requirements-dev.txt
pip install -e .  # 開発モードインストール
```

### 2. 課題の選択
- GitHub Issuesで適切な課題を探す
- 既存のIssueに取り組むか、新しいIssueを作成
- 複雑すぎる課題は小さなタスクに分解

### 3. ブランチの作成
```bash
# Issue番号に基づいたブランチ名
git checkout -b feature/issue-123-new-feature
git checkout -b fix/issue-456-bug-fix
git checkout -b docs/update-api-reference

# ブランチ名の規約
# - feature/機能名: 新機能
# - fix/バグ名: バグ修正
# - docs/ドキュメント名: ドキュメント更新
# - refactor/リファクタリング名: リファクタリング
# - test/テスト名: テスト関連
```

### 4. 開発
- コードを実装
- テストを追加
- ドキュメントを更新
- コーディング標準に従う

### 5. テスト
```bash
# すべてのテストを実行
pytest

# 新しい機能のテストのみ実行
pytest tests/ -k "new_feature"

# カバレッジレポートを生成
pytest --cov=tumblr_image_collector --cov-report=html
```

### 6. プルリクエストの作成
- 変更内容をコミット
- プルリクエストを作成
- レビューの待機

## コーディング標準 (Coding Standards)

### Python標準 (Python Standards)
- **PEP 8**: Pythonコーディング標準
- **PEP 484**: 型ヒント
- **PEP 526**: 変数アノテーション

### コードスタイル (Code Style)
```python
# 良い例
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """画像処理を行うクラス

    このクラスは、メモリ効率を考慮した画像処理を提供します。

    Attributes:
        max_dimension: 最大画像サイズ
        quality_threshold: 品質閾値
    """

    def __init__(self, max_dimension: int = 2048, quality_threshold: float = 0.7):
        """初期化

        Args:
            max_dimension: 最大画像サイズ
            quality_threshold: 品質判定の閾値

        Raises:
            ValueError: 無効なパラメータの場合
        """
        if max_dimension <= 0:
            raise ValueError("max_dimension must be positive")

        self.max_dimension = max_dimension
        self.quality_threshold = quality_threshold

    def process_image(self, image_path: Path) -> Optional[Dict[str, Any]]:
        """画像を処理する

        Args:
            image_path: 処理する画像のパス

        Returns:
            処理結果、またはNone（失敗時）

        Raises:
            FileNotFoundError: 画像ファイルが存在しない場合
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            # 処理の実装
            result = self._process_internal(image_path)
            logger.info(f"Successfully processed: {image_path}")
            return result
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

    def _process_internal(self, image_path: Path) -> Dict[str, Any]:
        """内部処理メソッド（プライベート）"""
        # 実装の詳細
        return {"status": "processed"}
```

### 禁止事項 (Prohibited Practices)
- ハードコードされた認証情報
- グローバル変数の多用
- 深いネスト（3レベル以上）
- 長い関数（50行以上）
- 複雑な条件分岐
- 例外の無視
- TODOコメントの残存

## テスト (Testing)

### テストの種類 (Test Types)
1. **単体テスト**: 個別の関数/メソッド
2. **統合テスト**: 複数のコンポーネントの連携
3. **エンドツーエンドテスト**: 完全なワークフロー
4. **パフォーマンステスト**: 速度とメモリ使用量
5. **セキュリティテスト**: 脆弱性チェック

### テストの書き方 (Writing Tests)
```python
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

class TestImageProcessor(unittest.TestCase):
    """ImageProcessorのテスト"""

    def setUp(self):
        """テスト前のセットアップ"""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = ImageProcessor()

    def tearDown(self):
        """テスト後のクリーンアップ"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_process_valid_image(self):
        """有効な画像の処理テスト"""
        # テストデータの準備
        image_path = Path(self.temp_dir) / "test.jpg"
        self._create_test_image(image_path)

        # 処理の実行
        result = self.processor.process_image(image_path)

        # 結果の検証
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "processed")

    def test_process_nonexistent_image(self):
        """存在しない画像の処理テスト"""
        nonexistent_path = Path(self.temp_dir) / "nonexistent.jpg"

        # 例外の検証
        with self.assertRaises(FileNotFoundError):
            self.processor.process_image(nonexistent_path)

    @patch('pathlib.Path.exists')
    def test_process_image_with_permission_error(self, mock_exists):
        """権限エラーのテスト"""
        mock_exists.return_value = False

        with self.assertRaises(FileNotFoundError):
            self.processor.process_image("test.jpg")

    def _create_test_image(self, path: Path):
        """テスト用の画像を作成（ヘルパーメソッド）"""
        from PIL import Image
        image = Image.new('RGB', (100, 100), color='red')
        image.save(path)
```

### テストカバレッジ (Test Coverage)
- **目標**: 90%以上のカバレッジ
- **必須**: すべての新しいコードにテスト
- **推奨**: エッジケースのテスト
- **定期**: カバレッジレポートの確認

## ドキュメント (Documentation)

### ドキュメントの更新
- すべての新しい機能にドキュメント
- API変更時のドキュメント更新
- コード例の提供
- スクリーンショットの追加（UI関連の場合）

### ドキュメントの種類 (Documentation Types)
1. **APIドキュメント**: 関数/クラスの詳細
2. **ユーザーガイド**: 使用方法の説明
3. **開発者ガイド**: 開発プロセスの説明
4. **チュートリアル**: ステップバイステップのガイド
5. **FAQ**: よくある質問と回答

### ドキュメントの書き方 (Writing Documentation)
```python
def complex_algorithm(data: List[int], threshold: float = 0.5) -> Dict[str, Any]:
    """
    複雑なアルゴリズムを実行する

    この関数は、入力データを処理し、統計情報を計算します。
    アルゴリズムの詳細な説明...

    Args:
        data: 処理するデータのリスト
        threshold: 判定の閾値（デフォルト: 0.5）

    Returns:
        計算結果の辞書
        - 'mean': データの平均値
        - 'median': データの中央値
        - 'filtered': 閾値以上のデータ

    Examples:
        >>> result = complex_algorithm([1, 2, 3, 4, 5])
        >>> result['mean']
        3.0

        >>> result = complex_algorithm([1, 2, 3, 4, 5], threshold=3.5)
        >>> result['filtered']
        [4, 5]

    Raises:
        ValueError: データが空の場合
        TypeError: データ型が不正の場合

    Notes:
        計算量: O(n log n)
        メモリ使用量: O(n)
    """
    if not data:
        raise ValueError("Data cannot be empty")

    # 実装
    return {
        'mean': sum(data) / len(data),
        'median': sorted(data)[len(data) // 2],
        'filtered': [x for x in data if x >= threshold]
    }
```

## プルリクエスト (Pull Requests)

### プルリクエストの作成
1. ブランチが最新の状態であることを確認
2. 変更内容を適切にコミット
3. プルリクエストテンプレートを使用
4. 適切なタイトルと説明を記述

### プルリクエストテンプレート
```markdown
## 変更の種類 (Type of Change)
- [ ] Bug fix (バグ修正)
- [ ] New feature (新機能)
- [ ] Breaking change (破壊的変更)
- [ ] Documentation (ドキュメント)
- [ ] Refactoring (リファクタリング)
- [ ] Performance improvement (パフォーマンス改善)

## 変更の説明 (Description)
<!-- 変更内容の詳細な説明 -->

## 関連するIssue (Related Issues)
<!-- 関連するIssue番号を記述 -->
Fixes #123

## 変更の確認方法 (How to Verify)
<!-- 変更を確認する方法を記述 -->
1. 特定のコマンドを実行
2. 特定のテストを実行
3. 特定の動作を確認

## テストの確認 (Testing Checklist)
- [ ] 既存のテストがすべて通る
- [ ] 新しいテストを追加
- [ ] 手動テストを実施
- [ ] エッジケースをテスト

## スクリーンショット (Screenshots)
<!-- UI変更の場合、スクリーンショットを追加 -->

## 追加のコメント (Additional Comments)
<!-- その他のコメント -->
```

### コミットメッセージのガイドライン
```bash
# 良いコミットメッセージ
git commit -m "Add user authentication feature

- Implement OAuth2 login flow
- Add user session management
- Update API endpoints for user data
- Add comprehensive tests

Fixes #456"

# 悪いコミットメッセージ（避ける）
git commit -m "Update code"
git commit -m "Fix bug"
git commit -m "Add stuff"
```

## レビュープロセス (Review Process)

### レビューのポイント (Review Points)
1. **コード品質**: コーディング標準への準拠
2. **機能性**: 機能が正しく動作するか
3. **テスト**: 適切なテストがあるか
4. **ドキュメント**: ドキュメントが更新されているか
5. **セキュリティ**: セキュリティ上の問題がないか
6. **パフォーマンス**: パフォーマンスへの影響

### レビューチェックリスト (Review Checklist)
- [ ] コードスタイルが正しい
- [ ] 型ヒントが適切に付与されている
- [ ] テストが十分にある
- [ ] ドキュメントが更新されている
- [ ] エラーハンドリングが適切
- [ ] セキュリティ上の問題がない
- [ ] パフォーマンスに悪影響がない
- [ ] 既存の機能に影響がない

### レビューのレスポンス
- 建設的なフィードバックを提供
- 具体的な改善点を提案
- 承認または修正依頼の判断

## コミュニティ (Community)

### コミュニティガイドライン (Community Guidelines)
- 敬意を持って接する
- 建設的な議論を行う
- 質問は歓迎する
- フィードバックを積極的に受け入れる
- 知識を共有する

### コミュニティリソース (Community Resources)
- **GitHub Issues**: バグ報告と機能リクエスト
- **ディスカッション**: 一般的な議論
- **Wiki**: プロジェクト情報の共有
- **チャット**: リアルタイムのコミュニケーション

### メンターシップ (Mentorship)
- 新しいコントリビューターを歓迎
- 質問に丁寧に答える
- コードレビューの機会を提供
- ベストプラクティスを共有

## 品質基準 (Quality Standards)

### コード品質 (Code Quality)
- **複雑度**: 関数あたりの複雑度を10以下に保つ
- **重複**: コードの重複を避ける
- **可読性**: コードが理解しやすいこと
- **保守性**: 変更が容易であること

### テスト品質 (Test Quality)
- **カバレッジ**: 90%以上のテストカバレッジ
- **信頼性**: テストが安定して通る
- **速度**: テストの実行時間が短い
- **保守性**: テストコードの保守が容易

### ドキュメント品質 (Documentation Quality)
- **正確性**: 正確な情報提供
- **完全性**: 必要な情報がすべて含まれている
- **最新性**: 常に最新の情報
- **使いやすさ**: 利用者が理解しやすい

## コントリビューター向けリソース (Contributor Resources)

### 学習リソース (Learning Resources)
- [Python公式ドキュメント](https://docs.python.org/)
- [pytestドキュメント](https://pytest.org/)
- [Blackコードフォーマッタ](https://black.readthedocs.io/)
- [MyPy型チェッカー](https://mypy.readthedocs.io/)

### 開発ツール (Development Tools)
- **IDE**: VSCode, PyCharm, Vim, Emacs
- **デバッガー**: pdb, ipdb, debugpy
- **プロファイラー**: cProfile, line_profiler, memory_profiler
- **リンター**: flake8, bandit, mypy

### コミュニティリソース (Community Resources)
- [Pythonコミュニティ](https://www.python.org/community/)
- [GitHubコミュニティガイドライン](https://docs.github.com/en/github/site-policy/github-community-guidelines)
- [オープンソースガイド](https://opensource.guide/)

## 謝辞 (Acknowledgements)

貢献してくださるすべての皆様に感謝します。あなたの貢献がTumblr Image Collectorをより良いプロジェクトにしています。

### コントリビューターの認識 (Contributor Recognition)
- 貢献の大きさに応じてクレジットを提供
- 特別な貢献に対してはメンション
- コミュニティでの貢献を称賛

---

**貢献ガイドライン v2.0.0** - Tumblr Image Collector
