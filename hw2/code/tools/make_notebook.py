import ast
import hashlib
import json
from pathlib import Path
import pprint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECTIONS = {'cells': [{'cell_type': 'markdown',
            'metadata': {},
            'source': ['# HW2 · Super-Convergence на Kaggle\n',
                       '[Статья](https://arxiv.org/abs/1708.07120) · [авторский '
                       'код](https://github.com/lnsmith54/super-convergence)\n',
                       '\n',
                       'Самостоятельный notebook для нашей адаптации эксперимента: один LR-пилот и три '
                       'режима для seeds **42, 43, 44**. Исходники встроены. Добавьте **CIFAR-10 Python** '
                       '(`pankrzysiu/cifar10-python`) через Add Input, включите GPU и выберите **Save '
                       'Version → Save & Run All**.\n',
                       '\n',
                       'LR выбирается один раз по validation пилота seed 42 и фиксируется для всех повторов. '
                       'Test не участвует в подборе. Уже завершённые прогоны с теми же настройками '
                       'пропускаются; незавершённые продолжаются с checkpoint. На исходных запусках max LR '
                       'составил **1.0**; новые результаты следует сохранять отдельно от представленных в '
                       'отчёте.\n'],
            'id': 'hw2-000'},
           {'cell_type': 'code',
            'metadata': {},
            'source': ['from pathlib import Path\n',
                       'import json, os, sys, subprocess, shutil\n',
                       'import torch\n',
                       '\n',
                       "WORKDIR = Path('/kaggle/working/hw2')\n",
                       "CIFAR_INPUTS = sorted(Path('/kaggle/input').rglob('cifar-10-batches-py'))\n",
                       'if not CIFAR_INPUTS:\n',
                       "    raise RuntimeError('Attach the CIFAR-10 Python dataset through Kaggle Input "
                       "first.')\n",
                       'DATA_ROOT = CIFAR_INPUTS[0].parent\n',
                       "print('CIFAR-10 source:', DATA_ROOT)\n",
                       'SEEDS = [42, 43, 44]\n',
                       "EXPERIMENTS = ['baseline_long', 'baseline_short', 'onecycle']\n",
                       '\n',
                       'STOP_AFTER_EPOCH = None\n',
                       '\n',
                       'if not torch.cuda.is_available():\n',
                       "    raise RuntimeError('Включите GPU в настройках Kaggle перед обучением.')\n",
                       "print('Python:', sys.version.split()[0], 'PyTorch:', torch.__version__)\n",
                       "print('GPU:', torch.cuda.get_device_name(0), 'CUDA:', torch.version.cuda)\n"],
            'execution_count': None,
            'outputs': [],
            'id': 'hw2-001'},
           {'cell_type': 'markdown',
            'metadata': {},
            'source': ['## Исходники и окружение\n',
                       'Для продолжения прошлой сессии сначала скопируйте её сохранённую папку `hw2` из '
                       'Kaggle Input в `WORKDIR`. Существующий конфиг сохранится. Если исходники отличаются '
                       'от встроенных, notebook остановится, чтобы не смешать версии.\n'],
            'id': 'hw2-002'},
           {'cell_type': 'code',
            'metadata': {},
            'source': ['os.chdir(WORKDIR)\n',
                       'import torchvision, numpy, matplotlib\n',
                       "print('torchvision:', torchvision.__version__, 'numpy:', numpy.__version__, "
                       "'matplotlib:', matplotlib.__version__)\n",
                       '\n',
                       'def run_experiment(name, seed, stop_after=None):\n',
                       "    config = json.loads((WORKDIR / 'configs/kaggle.json').read_text())\n",
                       "    expected = {k:v for k,v in config.items() if k not in ('experiments', "
                       "'lr_range')}\n",
                       "    expected.update(config['lr_range'] if name == 'lr_range' else "
                       "config['experiments'][name])\n",
                       "    output = WORKDIR / 'results' / f'{name}_seed{seed}'\n",
                       "    if (output / 'metadata.json').exists():\n",
                       "        previous = json.loads((output / 'metadata.json').read_text())\n",
                       "        if previous['config'] != expected:\n",
                       "            raise RuntimeError(f'Настройки {name} изменились. Выберите новую "
                       "WORKDIR.')\n",
                       "    if (output / 'summary.json').exists():\n",
                       "        print('Готовый прогон:', name, seed)\n",
                       '        return\n',
                       "    command = [sys.executable, '-m', 'src.train', '--experiment', name, '--seed', "
                       'str(seed),\n',
                       "               '--data-root', str(DATA_ROOT), '--device', 'cuda:0']\n",
                       "    if (output / 'last.pt').exists():\n",
                       "        command += ['--resume']\n",
                       '    elif output.exists() and any(output.iterdir()):\n',
                       "        raise RuntimeError(f'Незавершённый запуск без checkpoint: {output}. "
                       "Сохраните логи и используйте другую папку.')\n",
                       '    if stop_after is not None:\n',
                       "        command += ['--stop-after-epoch', str(stop_after)]\n",
                       '    subprocess.run(command, check=True)\n'],
            'execution_count': None,
            'outputs': [],
            'id': 'hw2-003'},
           {'cell_type': 'markdown',
            'metadata': {},
            'source': ['## 1. LR range test\n',
                       'Линейный рост LR от 0.01 до 2.0 за 1000 шагов. Validation измеряется каждые 50 '
                       'шагов. При расходимости тест остановится. Его веса не переходят в основные '
                       'эксперименты.\n'],
            'id': 'hw2-004'},
           {'cell_type': 'code',
            'metadata': {},
            'source': ["run_experiment('lr_range', 42)\n",
                       "subprocess.run([sys.executable, '-m', 'src.analyze'], check=True)\n",
                       'from IPython.display import display, Image, FileLink, Markdown\n',
                       "display(Image(filename=str(WORKDIR / 'results/figures/lr_range_seed42.png')))\n"],
            'execution_count': None,
            'outputs': [],
            'id': 'hw2-005'},
           {'cell_type': 'markdown',
            'metadata': {},
            'source': ['## 2. Выбор LR по отдельному пилоту\n',
                       'Правило задано до основного обучения: найти лучшую validation accuracy среди '
                       'измерений с LR ≥ 0.2; при равенстве выбрать меньший LR. Взять 80% этого LR и '
                       'ограничить максимум диапазоном [0.2, 1.0]. Это эвристика, а не гарантия '
                       'super-convergence. Выбор сохраняется в `lr_selection.json` и используется для всех '
                       'трёх seeds без повторного подбора.\n'],
            'id': 'hw2-006'},
           {'cell_type': 'code',
            'metadata': {},
            'source': ['import csv, math\n',
                       '\n',
                       '\n',
                       '\n',
                       '\n',
                       "range_path = WORKDIR / 'results/lr_range_seed42/lr_range.csv'\n",
                       'with range_path.open() as file:\n',
                       '    pilot_rows = list(csv.DictReader(file))\n',
                       "candidates = [r for r in pilot_rows if r['val_accuracy'] and float(r['lr']) >= 0.2\n",
                       "              and math.isfinite(float(r['val_accuracy']))]\n",
                       'if not candidates:\n',
                       "    raise RuntimeError('No usable validation measurements in the LR pilot. Inspect "
                       "its logs.')\n",
                       "pilot_best = max(candidates, key=lambda r: (float(r['val_accuracy']), "
                       "-float(r['lr'])))\n",
                       "CHOSEN_MAX_LR = round(min(1.0, max(0.2, 0.8 * float(pilot_best['lr']))), 4)\n",
                       'LR_SELECTION_NOTE = (\n',
                       "    'Fixed before the main runs: best validation point of the separate LR range "
                       "test, '\n",
                       '    f"lr={pilot_best[\'lr\']}, val_accuracy={pilot_best[\'val_accuracy\']}. "\n',
                       "    'Chosen maximum is 80% of this LR, bounded to [0.2, 1.0]. '\n",
                       "    'This is a conservative heuristic, not a guarantee of super-convergence.'\n",
                       ')\n',
                       "config_path = WORKDIR / 'configs/kaggle.json'\n",
                       'config = json.loads(config_path.read_text())\n',
                       "for name in ['onecycle', 'onecycle_high_wd']:\n",
                       "    config['experiments'][name]['lr_max'] = CHOSEN_MAX_LR\n",
                       "    config['experiments'][name]['lr_start'] = CHOSEN_MAX_LR / 10\n",
                       "    config['experiments'][name]['lr_final'] = CHOSEN_MAX_LR / 10000\n",
                       'for name in EXPERIMENTS:\n',
                       '    for seed in SEEDS:\n',
                       "        path = WORKDIR / 'results' / f'{name}_seed{seed}' / 'metadata.json'\n",
                       '        if path.exists():\n',
                       '            prior = json.loads(path.read_text())\n',
                       "            if prior['config']['schedule'] == 'onecycle' and "
                       "prior['config']['lr_max'] != config['experiments'][name]['lr_max']:\n",
                       "                raise ValueError('LR differs from a started experiment. Use a fresh "
                       "WORKDIR.')\n",
                       'config_path.write_text(json.dumps(config, indent=2))\n',
                       "(WORKDIR / 'lr_selection.json').write_text(json.dumps(\n",
                       "    {'max_lr':CHOSEN_MAX_LR, 'reason':LR_SELECTION_NOTE, 'pilot_best':pilot_best}, "
                       'indent=2))\n',
                       "print('Selected maximum LR:', CHOSEN_MAX_LR)\n",
                       'print(LR_SELECTION_NOTE)\n'],
            'execution_count': None,
            'outputs': [],
            'id': 'hw2-007'},
           {'cell_type': 'markdown',
            'metadata': {},
            'source': ['## 3. Основное сравнение\n',
                       '`baseline_long`: 120 эпох, multistep. `baseline_short`: 24 эпохи с тем же бюджетом, '
                       'что и 1cycle. `onecycle`: 24 эпохи, три линейные фазы LR и обратное изменение '
                       'momentum.\n',
                       '\n',
                       'Train/validation/test: 45k/5k/10k, одинаковое разделение. Каждый метод начинается с '
                       'одинаковой инициализации для данного seed. Все методы используют batch 512 и '
                       'BatchNorm momentum 0.1. Лучший checkpoint выбирается по validation. Test оценивается '
                       'после завершения прогона.\n',
                       '\n',
                       'Все три seeds выполняются последовательно с одним фиксированным LR. Оцените время по '
                       'логам первых эпох. Для сохранения частичного запуска задайте `STOP_AFTER_EPOCH`, '
                       'скачайте Output, затем продолжите. Цена LR range test не входит во время основного '
                       'обучения и обсуждается отдельно.\n'],
            'id': 'hw2-008'},
           {'cell_type': 'code',
            'metadata': {},
            'source': ['for seed in SEEDS:\n',
                       '    for experiment in EXPERIMENTS:\n',
                       '        run_experiment(experiment, seed, STOP_AFTER_EPOCH)\n'],
            'execution_count': None,
            'outputs': [],
            'id': 'hw2-009'},
           {'cell_type': 'markdown',
            'metadata': {},
            'source': ['## 4. Метрики и интерпретация\n',
                       'Сравниваем test accuracy и SD по seeds, парную разницу точности, время/шаги до 85% '
                       'validation accuracy. Порог 85% выбран для нашего анализа и не является критерием '
                       'статьи. Статистическая эквивалентность методов не установлена.\n',
                       '\n',
                       'Отношение бюджетов 120/24 = 5 само по себе не доказывает ускорение. Если порог не '
                       'достигнут, время до него равно null. Графики относятся к validation. Train accuracy '
                       'измеряется на аугментированных батчах во время обновления весов, поэтому её разница '
                       'с validation не является точной оценкой generalization gap.\n'],
            'id': 'hw2-010'},
           {'cell_type': 'code',
            'metadata': {},
            'source': ["subprocess.run([sys.executable, '-m', 'src.analyze'], check=True)\n",
                       "report = WORKDIR / 'results/results.md'\n",
                       'if report.exists():\n',
                       '    display(Markdown(report.read_text()))\n',
                       "chart = WORKDIR / 'results/figures/convergence.png'\n",
                       'if chart.exists():\n',
                       '    display(Image(filename=str(chart)))\n'],
            'execution_count': None,
            'outputs': [],
            'id': 'hw2-011'},
           {'cell_type': 'markdown',
            'metadata': {},
            'source': ['## 5. Сохранение\n',
                       'Архив включает checkpoints для продолжения, настройки и сырые метрики. Скачайте '
                       'архив и сохраните версию notebook с Output. Для GitHub достаточно исходников, '
                       'выбранных логов/графиков, финальной презентации и отчёта PDF. CIFAR-10 и checkpoints '
                       'публиковать необязательно.\n'],
            'id': 'hw2-012'},
           {'cell_type': 'code',
            'metadata': {},
            'source': ["archive_base = Path('/kaggle/working/hw2_kaggle_results')\n",
                       "archive_path = shutil.make_archive(str(archive_base), 'zip', "
                       'root_dir=WORKDIR.parent, base_dir=WORKDIR.name)\n',
                       'display(FileLink(archive_path))\n',
                       "print('Сохранено:', archive_path)\n"],
            'execution_count': None,
            'outputs': [],
            'id': 'hw2-013'}],
 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
              'language_info': {'name': 'python'}},
 'nbformat': 4,
 'nbformat_minor': 5}


def build(template, destination):
    notebook = json.loads(template.read_text()) if template is not None else json.loads(json.dumps(DEFAULT_SECTIONS))
    cells = []
    for cell in notebook['cells']:
        if cell['cell_type'] != 'markdown':
            continue
        text = ''.join(cell['source'])
        text = text.replace('полного воспроизведения', 'нашей адаптации эксперимента')
        text = text.replace('тем же протоколом', 'теми же настройками')
        text = text.replace('Допуск 1 процентный пункт для обсуждения сопоставимости является описательным, без теста эквивалентности.', 'Порог 85% выбран для нашего анализа и не является критерием статьи. Статистическая эквивалентность методов не установлена.')
        if text.startswith('## 1. LR range test'):
            text += '\nИсточник идеи: статья, раздел 3. Границы 0.01–2.0, 1000 шагов и проверка validation каждые 50 шагов выбраны для нашего запуска; это не значения из статьи.\n'
        if text.startswith('## 2. Выбор LR'):
            text += '\nПравило 80% и ограничение [0.2, 1.0] отсутствуют в статье. В разделе 3 авторы описывают выбор максимума по пику test accuracy в LR range test; наш код использует отдельный validation и другое правило.\n'
        if text.startswith('## 3. Основное сравнение'):
            text += '\nЭти эпохи, разделение данных и выбор checkpoint относятся к нашей адаптации. Циклический momentum 0.95–0.85 встречается в статье для wide ResNet и CIFAR-100 (таблица 2), но не является настройкой основного CIFAR-10 / ResNet-56 примера (таблица 8: momentum 0.9).\n'
        text = text.replace('Same protocol', 'Same experiment settings').replace('## 1. Fixed protocol from seed 42', '## 1. Сохранённые настройки нашего seed 42')
        text = text.replace('Сначала один seed, затем 42/43/44.', 'В этой сохранённой версии последовательно запускаются seeds 43 и 44; seed 42 выполнен в версии 1.')
        cell['source'] = text.splitlines(keepends=True)

    def add(kind, text):
        cell = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    sources = {f'src/{name}.py': (ROOT / 'src' / f'{name}.py').read_text()
               for name in ['__init__', 'models', 'data', 'schedules', 'train', 'analyze']}
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()}
    setup = ''.join(notebook['cells'][1]['source'])
    setup += '\nWORKDIR.mkdir(parents=True, exist_ok=True)\n(WORKDIR / "src").mkdir(exist_ok=True)\n'
    setup += 'SOURCE_SHA256 = ' + pprint.pformat(hashes, width=100, sort_dicts=False) + '\n'
    setup += '''for name, expected_hash in SOURCE_SHA256.items():
    path = WORKDIR / name
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError(f'Отличается версия {name}. Используйте новую WORKDIR.')
(WORKDIR / 'src/__init__.py').write_text('\n')
os.chdir(WORKDIR)
'''
    setup = setup.replace('import json, os, sys, subprocess, shutil', 'import json, os, sys, subprocess, shutil, hashlib')
    setup = setup.replace("write_text('\n')", "write_text('\\n')")
    add('markdown', ''.join(notebook['cells'][0]['source']))
    add('code', setup)
    add('markdown', '''## Статья и выполненная адаптация
Объяснение метода основано на [статье, версия v3](https://arxiv.org/abs/1708.07120v3). Этот notebook сохраняет настройки уже выполненных запусков. Он не является точным повторением основного примера авторов.

| Компонент | Статья: CIFAR-10 / ResNet-56 | Наш запуск |
|---|---|---|
| Реализация | Caffe, приложение A.4 | PyTorch, Tesla T4 |
| Данные для обучения | 50 000, таблица 6 | 45 000 train / 5 000 validation |
| Бюджеты | около 80 000 / 10 000 шагов, рисунок 1a | 10 560 / 2 112 шагов (120 / 24 эпохи) |
| Batch | 1000 для основного примера; 512 также исследован в таблице 7 | 512 |
| Большой LR | 0.1–3.0, таблица 6 | 0.1–1.0, выбран по нашему пилоту |
| Momentum | 0.9 для основного примера, таблица 8 | 0.95–0.85–0.95 для 1cycle |
| Уменьшение LR | один цикл и завершение с малым LR, раздел 3 | фазы 45%/45%/10%, конечный LR 0.0001 |
| Сравнение скорости | шаги и test accuracy, рисунок 1a | дополнительно время до 85% validation |

Seeds 42/43/44, выделенный validation, правило выбора LR и порог 85% заданы в нашей реализации; статья не задаёт их для этого сравнения. Результаты notebook являются нашими измерениями и не подменяют опубликованные результаты.
''')
    add('markdown', '## Настройки нашего запуска\nЭто настройки выполненной адаптации для Kaggle, а не конфигурация основного эксперимента статьи. При продолжении запуска ранее выбранные значения сохраняются.\n')
    config = json.loads((ROOT / 'configs/kaggle.json').read_text())
    add('code', 'DEFAULT_CONFIG = ' + pprint.pformat(config, width=100, sort_dicts=False) + '''
config_path = WORKDIR / 'configs/kaggle.json'
config_path.parent.mkdir(exist_ok=True)
if not config_path.exists():
    config_path.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
''')
    headings = {
        'models': 'Модель ResNet-56',
        'data': 'Данные CIFAR-10',
        'schedules': 'Изменение learning rate и momentum в ходе обучения',
        'train': 'Обучение и оценка',
        'analyze': 'Анализ результатов',
    }
    for name, heading in headings.items():
        add('markdown', f'## {heading}\n')
        add('code', f'%%writefile {{WORKDIR}}/src/{name}.py\n' + sources[f'src/{name}.py'])
    payload_cell = ''.join(notebook['cells'][3]['source'])
    helpers = payload_cell[payload_cell.index('os.chdir(WORKDIR)'):]
    add('markdown', '## Запуск отдельных экспериментов\n')
    add('code', helpers)
    for cell in notebook['cells'][4:]:
        add(cell['cell_type'], ''.join(cell['source']))
    for i, cell in enumerate(cells):
        cell['id'] = f'hw2-{i:03d}'
        if cell['cell_type'] == 'code':
            text = ''.join(cell['source'])
            ast.parse(text.split('\n', 1)[1] if text.startswith('%%writefile') else text)
    notebook['cells'] = cells
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    build(None, ROOT.parent / 'kaggle_super_convergence.ipynb')
    print('Created notebook with readable Python cells.')
