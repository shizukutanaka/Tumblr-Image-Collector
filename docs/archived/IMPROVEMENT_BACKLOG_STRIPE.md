# Stripe課金および製品改善バックログ / Stripe Billing and Product Improvement Backlog

## 概要 / Overview
本書はStripe課金連携とTumblr Image Collector全体の改善案を網羅的に列挙し、優先度および対象領域ごとに整理します。利用者視点での利便性向上と運用者視点での保守性向上の両立を目的とします。
This document enumerates comprehensive improvement opportunities for Stripe billing integration and the overall Tumblr Image Collector. It balances user-facing convenience with operator-focused maintainability.

## 優先度概要 / Priority Synopsis
- **高優先度 High Priority**: 直近スプリントで着手し、Stripe課金の信頼性とユーザー体験を左右する事項。
- **中優先度 Medium Priority**: 高優先度の裏付けや運用効率化につながる事項。
- **低優先度 Low Priority**: 長期的価値を生む最適化。高・中優先度完了後に検討。

## 改善項目一覧 / Improvement Catalogue
1. **[ID:001 高優先度 | High Priority]** 日本語: Stripeの`publishable_key`と`secret_key`を環境変数から安全に読み込む機構を追加する。 English: Add secure environment-variable loading for Stripe `publishable_key` and `secret_key`.
2. **[ID:002 高優先度 | High Priority]** 日本語: Stripe Webhook検証で`webhook_secret`を使用し署名検証を必須化する。 English: Enforce signature validation using Stripe `webhook_secret` for webhook verification.
3. **[ID:003 高優先度 | High Priority]** 日本語: `StripeBillingManager`でAPIキーをリクエストごとに設定しスレッド安全性を担保する。 English: Set Stripe API keys per request within `StripeBillingManager` to ensure thread safety.
4. **[ID:004 高優先度 | High Priority]** 日本語: サブスクリプション作成後に`stripe_subscription_id`を`LicenseManager`へ記録する処理を実装する。 English: Persist `stripe_subscription_id` in `LicenseManager` after subscription creation.
5. **[ID:005 高優先度 | High Priority]** 日本語: Checkoutセッション生成時に`success_url`へライセンス反映用トークンを付与する。 English: Append license application token to `success_url` during checkout session creation.
6. **[ID:006 高優先度 | High Priority]** 日本語: `cancel_url`アクセス時に再開案内を表示する専用ページを用意する。 English: Provide a dedicated page explaining how to resume when visiting the `cancel_url`.
7. **[ID:007 高優先度 | High Priority]** 日本語: サブスクリプションプラン`pro_monthly`の料金や説明文を翻訳済みで表示する。 English: Display localized pricing and description for the `pro_monthly` subscription plan.
8. **[ID:008 高優先度 | High Priority]** 日本語: Checkout作成エラーを`production_error_handler`経由で捕捉し自動リトライする。 English: Route checkout creation failures through `production_error_handler` for automated retry.
9. **[ID:009 高優先度 | High Priority]** 日本語: `config.json`にStripe設定が欠落している場合の初回起動時チェックを追加する。 English: Add first-launch validation for missing Stripe configuration in `config.json`.
10. **[ID:010 高優先度 | High Priority]** 日本語: Stripe APIバージョンを固定し互換性問題を防止する。 English: Pin the Stripe API version to avoid compatibility regressions.
11. **[ID:011 高優先度 | High Priority]** 日本語: Checkout完了後にブラウザで自動的に成功画面を開くハンドラを用意する。 English: Implement a handler that automatically opens the success page in a browser after checkout.
12. **[ID:012 高優先度 | High Priority]** 日本語: Webhookで`invoice.payment_succeeded`を受信しライセンスを自動更新する。 English: Handle `invoice.payment_succeeded` webhooks to refresh license status automatically.
13. **[ID:013 高優先度 | High Priority]** 日本語: Webhookで`customer.subscription.deleted`を受信しステータスを`EXPIRED`へ切り替える。 English: Update license status to `EXPIRED` upon receiving `customer.subscription.deleted` webhook.
14. **[ID:014 高優先度 | High Priority]** 日本語: Webhook処理の冪等性を確保するためイベントIDを記録する。 English: Persist webhook event IDs to maintain idempotency.
15. **[ID:015 高優先度 | High Priority]** 日本語: Checkout セッション生成APIのレスポンス時間を監視ダッシュボードに送信する。 English: Send checkout session creation latency metrics to the monitoring dashboard.
16. **[ID:016 高優先度 | High Priority]** 日本語: `LicenseManager`へプラン別機能フラグを導入しサブスクリプション特典を制御する。 English: Introduce plan-specific feature flags in `LicenseManager` to control subscription entitlements.
17. **[ID:017 高優先度 | High Priority]** 日本語: 自動課金に失敗した場合の通知メールテンプレートを整備する。 English: Prepare notification email templates for failed automatic billing attempts.
18. **[ID:018 高優先度 | High Priority]** 日本語: `tumblr_image_collector.py`のCLIにプラン一覧表示コマンドを追加する。 English: Add a CLI command in `tumblr_image_collector.py` to list available plans.
19. **[ID:019 高優先度 | High Priority]** 日本語: CLIからStripe Checkoutを起動できる対話フローを実装する。 English: Implement an interactive CLI flow that launches Stripe Checkout.
20. **[ID:020 高優先度 | High Priority]** 日本語: `config.py`のウィザードでStripeキー入力をサポートする質問項目を追加する。 English: Extend `config.py` wizard prompts to capture Stripe keys.
21. **[ID:021 高優先度 | High Priority]** 日本語: Stripe鍵入力時に形式検証とマスキング表示を行う。 English: Validate Stripe key format and mask input during wizard entry.
22. **[ID:022 高優先度 | High Priority]** 日本語: 成功画面でライセンス適用手順を日本語と英語で案内する。 English: Provide bilingual license application instructions on the success page.
23. **[ID:023 高優先度 | High Priority]** 日本語: `requirements.txt`へ`stripe`ライブラリを追加しバージョン管理する。 English: Add the `stripe` library to `requirements.txt` with explicit version control.
24. **[ID:024 高優先度 | High Priority]** 日本語: Stripe API通信でのネットワークタイムアウトを設定する。 English: Configure network timeouts for Stripe API communication.
25. **[ID:025 高優先度 | High Priority]** 日本語: WebhookエンドポイントのURLとシークレットを設定ファイルで切り替え可能にする。 English: Allow configuration-driven control of webhook endpoint URLs and secrets.
26. **[ID:026 高優先度 | High Priority]** 日本語: 課金イベントを監査ログに記録し操作追跡を可能にする。 English: Log billing events into the audit trail for traceability.
27. **[ID:027 高優先度 | High Priority]** 日本語: Checkout失敗時に再試行リンクとサポート連絡手段を提示する。 English: Display retry links and support contact options upon checkout failure.
28. **[ID:028 高優先度 | High Priority]** 日本語: `StripeBillingManager`で例外発生時に詳細なコンテキスト情報を記録する。 English: Record detailed context when exceptions occur inside `StripeBillingManager`.
29. **[ID:029 高優先度 | High Priority]** 日本語: サブスクリプションの請求周期をユーザーが選択できるUIを提供する。 English: Offer UI allowing users to choose subscription billing cycles.
30. **[ID:030 高優先度 | High Priority]** 日本語: `LicenseManager`保存ファイルの暗号化オプションを追加する。 English: Add optional encryption for `LicenseManager` storage files.
31. **[ID:031 高優先度 | High Priority]** 日本語: ライセンス情報が失われた際の復旧ガイドを同梱する。 English: Include a recovery guide for missing license data.
32. **[ID:032 高優先度 | High Priority]** 日本語: Stripe側でのプラン変更を自動検知しローカル設定を更新する。 English: Detect plan changes from Stripe and refresh local configuration automatically.
33. **[ID:033 高優先度 | High Priority]** 日本語: Checkoutリンクを有効期限付きにし不正利用を防止する。 English: Generate checkout links with expiration to prevent misuse.
34. **[ID:034 高優先度 | High Priority]** 日本語: Checkoutの成功・失敗イベントを`monitoring_system.py`でメトリクス化する。 English: Capture checkout success/failure metrics within `monitoring_system.py`.
35. **[ID:035 高優先度 | High Priority]** 日本語: サブスクリプション停止手続きの案内ページを整備する。 English: Provide a dedicated page explaining how to cancel subscriptions.
36. **[ID:036 高優先度 | High Priority]** 日本語: 月額プランの提供価値を明示するマーケティング文言を整える。 English: Clarify the value proposition of the monthly plan with concise marketing copy.
37. **[ID:037 高優先度 | High Priority]** 日本語: `personal_lifetime`プラン購入後のサポート導線を整備する。 English: Define post-purchase support flow for the `personal_lifetime` plan.
38. **[ID:038 高優先度 | High Priority]** 日本語: ライセンス情報が競合した場合のマージ戦略を実装する。 English: Implement conflict resolution strategy for license data merges.
39. **[ID:039 高優先度 | High Priority]** 日本語: Stripe API障害時にフォールバックライセンスキーを発行する仕組みを用意する。 English: Provide fallback license key issuance when Stripe API is unavailable.
40. **[ID:040 高優先度 | High Priority]** 日本語: CheckoutフローのA/Bテストを行える設定フラグを追加する。 English: Introduce feature flags enabling A/B testing of the checkout flow.
41. **[ID:041 高優先度 | High Priority]** 日本語: 定期課金の請求履歴をユーザーがダウンロードできるようにする。 English: Allow users to download billing history for subscriptions.
42. **[ID:042 高優先度 | High Priority]** 日本語: 返金リクエスト処理を自動化する管理ツールを用意する。 English: Automate refund request handling via an admin tool.
43. **[ID:043 高優先度 | High Priority]** 日本語: 課金成功後に`ui.py`で成功通知を表示する。 English: Display success notifications in `ui.py` after successful billing events.
44. **[ID:044 高優先度 | High Priority]** 日本語: ライセンス検証をアプリ起動時だけでなく定期的に行うスケジューラを導入する。 English: Schedule periodic license validation beyond application startup.
45. **[ID:045 高優先度 | High Priority]** 日本語: Stripeテストモードとライブモードを設定で切り替え可能にする。 English: Enable configuration-based toggling between Stripe test and live modes.
46. **[ID:046 高優先度 | High Priority]** 日本語: Stripe Webhook再送シナリオでの重複処理防止機構を実装する。 English: Implement duplication guards for Stripe webhook retries.
47. **[ID:047 高優先度 | High Priority]** 日本語: Checkout完了時に`LicenseManager.update_status`を直接呼び出すパスを整える。 English: Wire checkout completion to call `LicenseManager.update_status` directly.
48. **[ID:048 高優先度 | High Priority]** 日本語: ライセンス情報破損時の自己修復ロジックを実装する。 English: Implement self-healing logic for corrupt license data.
49. **[ID:049 高優先度 | High Priority]** 日本語: Stripeのレート制限に備えて指数バックオフ戦略を適用する。 English: Use exponential backoff strategies to respect Stripe rate limits.
50. **[ID:050 高優先度 | High Priority]** 日本語: `StripeBillingManager.list_products`でプランの表示順序を設定可能にする。 English: Allow configuring plan display order in `StripeBillingManager.list_products`.
51. **[ID:051 高優先度 | High Priority]** 日本語: サブスクリプションの試用期間設定をプランごとに管理する。 English: Manage trial periods per subscription plan.
52. **[ID:052 高優先度 | High Priority]** 日本語: Checkout成功後に生成されるライセンスファイル名をユーザーが指定できるようにする。 English: Allow users to specify license filename generated after checkout success.
53. **[ID:053 高優先度 | High Priority]** 日本語: 複数端末でのライセンス利用を制御するデバイス管理機能を追加する。 English: Add device management to control multi-device license usage.
54. **[ID:054 高優先度 | High Priority]** 日本語: Stripe顧客IDとローカルユーザーIDを紐付ける永続ストアを整備する。 English: Maintain mapping between Stripe customer IDs and local user IDs in persistent storage.
55. **[ID:055 高優先度 | High Priority]** 日本語: サブスクリプション更新処理を`production_monitoring.py`で監視する。 English: Monitor subscription renewal processes via `production_monitoring.py`.
56. **[ID:056 高優先度 | High Priority]** 日本語: ライセンス切れ近辺で自動通知メールを送信するスケジュールを組み込む。 English: Schedule automated email reminders for upcoming license expirations.
57. **[ID:057 高優先度 | High Priority]** 日本語: Stripeダッシュボードとの整合性チェックを自動実行する。 English: Automate reconciliations between local records and the Stripe dashboard.
58. **[ID:058 高優先度 | High Priority]** 日本語: `config_personal.json`でもStripe設定を上書き可能にする。 English: Allow overriding Stripe configuration within `config_personal.json`.
59. **[ID:059 高優先度 | High Priority]** 日本語: 月額プランのアップセルメッセージを`ui.py`に表示する。 English: Show monthly plan upsell messages within `ui.py`.
60. **[ID:060 高優先度 | High Priority]** 日本語: Checkout中の中断や通信失敗に備えたリカバリーガイドを提供する。 English: Provide a recovery guide for interruptions or network failures during checkout.
61. **[ID:061 高優先度 | High Priority]** 日本語: サブスクリプションが停止した場合に利用機能を段階的に制限する。 English: Gradually restrict feature access when a subscription lapses.
62. **[ID:062 高優先度 | High Priority]** 日本語: ライセンス適用に必要な最低限のAPIスコープを明確化する。 English: Document minimal API scopes required for license application.
63. **[ID:063 高優先度 | High Priority]** 日本語: 企業向け請求書払いプランをStripe Billingで提供する。 English: Offer invoice-based enterprise plans via Stripe Billing.
64. **[ID:064 高優先度 | High Priority]** 日本語: Webhookの受信ステータスを監視ダッシュボードで可視化する。 English: Visualize webhook reception status within the monitoring dashboard.
65. **[ID:065 高優先度 | High Priority]** 日本語: Checkout前に利用規約への同意を確認するフローを追加する。 English: Introduce a pre-checkout consent flow for terms of use.
66. **[ID:066 高優先度 | High Priority]** 日本語: `tumblr_image_collector.py`でStripe APIキーが設定されていない場合に警告を表示する。 English: Warn users in `tumblr_image_collector.py` when Stripe API keys are unset.
67. **[ID:067 高優先度 | High Priority]** 日本語: APIキー漏洩検知のため監査ログにアクセス履歴を記録する。 English: Record access logs to detect potential API key leakage.
68. **[ID:068 高優先度 | High Priority]** 日本語: ライセンス適用中にバックグラウンドタスクを一時停止する。 English: Pause background tasks while license updates are in progress.
69. **[ID:069 高優先度 | High Priority]** 日本語: Checkoutリンク生成時にユーザー属性と紐付ける。 English: Link user attributes when generating checkout URLs.
70. **[ID:070 高優先度 | High Priority]** 日本語: Stripeの`customer.portal`を有効化しユーザー自らプラン管理できるようにする。 English: Enable Stripe Customer Portal for self-service plan management.
71. **[ID:071 高優先度 | High Priority]** 日本語: Checkout成功後に即時でライセンスファイルを自動ダウンロードさせる。 English: Auto-download license files upon checkout success.
72. **[ID:072 高優先度 | High Priority]** 日本語: `LicenseManager`のステータス遷移をテストで保証する。 English: Cover `LicenseManager` state transitions with automated tests.
73. **[ID:073 高優先度 | High Priority]** 日本語: Stripe APIキーをローテーションする際の手順を文書化する。 English: Document steps for rotating Stripe API keys.
74. **[ID:074 高優先度 | High Priority]** 日本語: モバイル環境向けのレスポンシブなCheckout誘導ページを用意する。 English: Provide a mobile-responsive page guiding users to Checkout.
75. **[ID:075 高優先度 | High Priority]** 日本語: サブスクリプション利用状況を週次でメールレポートする機能を追加する。 English: Add weekly email reports summarizing subscription usage.
76. **[ID:076 高優先度 | High Priority]** 日本語: Stripe課金で得られる売上データをCSV出力できる管理機能を実装する。 English: Implement admin CSV export for Stripe revenue data.
77. **[ID:077 高優先度 | High Priority]** 日本語: ライセンス復旧のためのサポート問い合わせテンプレートを用意する。 English: Prepare support request templates for license recovery.
78. **[ID:078 高優先度 | High Priority]** 日本語: Stripe APIレスポンスを`logging_utils.py`でサニタイズして記録する。 English: Sanitize Stripe API responses before logging in `logging_utils.py`.
79. **[ID:079 高優先度 | High Priority]** 日本語: Checkoutセッションのキャンセル理由を収集し改善に活用する。 English: Collect checkout cancellation reasons for process improvements.
80. **[ID:080 高優先度 | High Priority]** 日本語: サブスクリプションの複数プラン切り替えをサポートするUIを整備する。 English: Provide UI supporting plan switches across multiple subscription tiers.
81. **[ID:081 高優先度 | High Priority]** 日本語: `stripe_billing.py`のJSONシリアライザでUnicode対応を強化する。 English: Enhance Unicode handling in `stripe_billing.py` JSON serializer.
82. **[ID:082 高優先度 | High Priority]** 日本語: Stripe課金にかかわる設定を`ConfigWizard`でテストモードと本番モードそれぞれ入力できるようにする。 English: Allow `ConfigWizard` to capture separate test and live Stripe settings.
83. **[ID:083 高優先度 | High Priority]** 日本語: ライセンスファイルに署名を施し改ざん検知を可能にする。 English: Sign license files to enable tamper detection.
84. **[ID:084 高優先度 | High Priority]** 日本語: ライセンスの状態が`TRIAL`から`ACTIVE`への遷移時に追加特典を通知する。 English: Notify users about unlocked benefits when transitioning from `TRIAL` to `ACTIVE`.
85. **[ID:085 高優先度 | High Priority]** 日本語: Checkout開始から完了までの操作をモニタリングし離脱率を可視化する。 English: Monitor the checkout funnel to visualize drop-off rates.
86. **[ID:086 高優先度 | High Priority]** 日本語: CLI利用者向けにCheckout URLをクリップボードへコピーする機能を提供する。 English: Offer clipboard copy of checkout URLs for CLI users.
87. **[ID:087 高優先度 | High Priority]** 日本語: サブスクリプションの利用可能機能一覧を`personal_features.py`で制御する。 English: Control feature availability per subscription in `personal_features.py`.
88. **[ID:088 高優先度 | High Priority]** 日本語: Webhook処理のテストを`tests/`配下に追加し通過基準を設定する。 English: Add webhook processing tests under `tests/` with clear pass criteria.
89. **[ID:089 高優先度 | High Priority]** 日本語: ライセンス更新時にアプリ内通知を発行する。 English: Emit in-app notifications upon license renewals.
90. **[ID:090 高優先度 | High Priority]** 日本語: Stripe課金に関連する障害発生時のオンコール手順をまとめる。 English: Compile on-call procedures for Stripe-related incidents.
91. **[ID:091 高優先度 | High Priority]** 日本語: サブスクリプションで得られる追加ストレージ容量を設定画面に表示する。 English: Display extra storage quota granted by the subscription in settings.
92. **[ID:092 高優先度 | High Priority]** 日本語: Checkoutリンクの生成履歴を`automation_analysis_report.json`に追跡する。 English: Track checkout link generation history in `automation_analysis_report.json`.
93. **[ID:093 高優先度 | High Priority]** 日本語: ライセンス失効時に自動で一部機能をデグレードさせる。 English: Automatically degrade select features upon license expiry.
94. **[ID:094 高優先度 | High Priority]** 日本語: Stripeイベント処理を非同期キューに載せシステム負荷を平準化する。 English: Offload Stripe event processing to an asynchronous queue.
95. **[ID:095 高優先度 | High Priority]** 日本語: Checkoutで失敗した決済手段を記録しサポート対応に備える。 English: Record failed payment methods for support readiness.
96. **[ID:096 高優先度 | High Priority]** 日本語: ライセンスファイルのバックアップを冗長化するオプションを追加する。 English: Introduce redundant backup options for license files.
97. **[ID:097 高優先度 | High Priority]** 日本語: `StripeBillingManager.from_config`でプラン定義の必須項目を検証する。 English: Validate required plan attributes within `StripeBillingManager.from_config`.
98. **[ID:098 高優先度 | High Priority]** 日本語: Checkout成功時にSlack通知を送るオプションを実装する。 English: Implement optional Slack notifications for successful checkouts.
99. **[ID:099 高優先度 | High Priority]** 日本語: ライセンス管理ディレクトリへのアクセス権を起動時にチェックする。 English: Check license storage directory permissions on startup.
100. **[ID:100 高優先度 | High Priority]** 日本語: サブスクリプションへのアップグレードパスをプロダクト内ヘルプに掲載する。 English: Document subscription upgrade paths within in-product help.
...
500. **[ID:500 低優先度 | Low Priority]** 日本語: 課金関連のコードに対して自動生成ドキュメントを整備しチーム共有を円滑にする。 English: Prepare auto-generated documentation for billing-related code to streamline team knowledge sharing.
