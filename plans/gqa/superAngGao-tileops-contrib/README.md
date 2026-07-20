# superAngGao 在 TileOPs 的贡献统计

统计时间：2026-05-05 22:55:08，仓库：tile-ai/TileOPs，口径：`upstream/main`。

## 可直接放 PPT 的结论

- 主线贡献：76 个 commits，占主线 535 个 commits 的 14.2%，作者排名第 2。
- PR 贡献：GitHub 上由 `superAngGao` 发起 91 个 PR，其中 76 个已合入，2 个仍打开。
- 代码规模：按主线 commit numstat 统计，新增 28,226 行、删除 6,766 行，合计 34,992 行变更。
- 高峰月份：2026-03，主线合入 49 个 commits。
- 主要类型：Fix 18, CI 14, Feat 12, Refactor 7, BugFix 7。
- 覆盖模块：tileops 17,221 行, benchmarks 6,247 行, tests 4,842 行, top 1,865 行, .github 1,811 行。

## PPT 图表文件

- `00_dashboard.png/svg`：一页总览仪表盘。
- `01_monthly_commits.png/svg`：月度主线提交趋势。
- `02_commit_type_mix.png/svg`：提交类型分布。
- `03_module_impact.png/svg`：模块影响范围。
- `04_pr_status.png/svg`：PR 状态统计。
- `superAngGao_tileops_contribution.pptx`：已排版好的 4 页 16:9 PPT 草稿。

## 口径说明

- Git 主线统计按 author email `gaoang0125@163.com` 归并，即 `Ang Gao <gaoang0125@163.com>`。
- PR 统计来自 GitHub CLI：`gh pr list -R tile-ai/TileOPs --author superAngGao --state all`。
- 行数统计来自 `git log upstream/main --author=gaoang0125@163.com --numstat`，适合展示贡献规模，不等同于当前仓库净增行数。
