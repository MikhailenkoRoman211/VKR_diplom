from __future__ import annotations

import threading
import datetime
import calendar
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
import matplotlib.figure as mpf
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import ttkbootstrap as tb  # pip install ttkbootstrap

from data_collector import DataCollector
from data_processor import DataProcessor
from model_manager import ModelManager
from optimizer import Optimizer

matplotlib.use("Agg")


class ToolTip:
    """Простой tooltip для любого виджета."""
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tipwindow: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        if self.tipwindow:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 1
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = ttk.Label(
            tw, text=self.text, justify="left",
            background="lightyellow", relief="solid", borderwidth=1
        )
        lbl.pack(ipadx=4, ipady=2)

    def _hide(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class CalendarDialog(tk.Toplevel):
    """Простой календарь для выбора даты, центрируется над окном-родителем."""
    def __init__(self, master, entry: ttk.Entry):
        super().__init__(master)
        self.entry = entry
        self.transient(master)
        self.title("Выберите дату")
        self.resizable(False, False)

        # Попытаемся взять текущую дату из поля, иначе — сегодня
        try:
            d = datetime.datetime.strptime(entry.get(), "%Y-%m-%d").date()
        except Exception:
            d = datetime.date.today()
        self.year, self.month = d.year, d.month

        # Навигация месяцев
        header = ttk.Frame(self)
        header.pack(padx=10, pady=5)
        ttk.Button(header, text="<", width=2, command=self._prev_month).pack(side="left")
        self.lbl_month = ttk.Label(header, text="")
        self.lbl_month.pack(side="left", padx=12)
        ttk.Button(header, text=">", width=2, command=self._next_month).pack(side="left")

        # Фрейм дней
        self.frm_days = ttk.Frame(self)
        self.frm_days.pack(padx=10, pady=5)

        self._build_calendar()

        # Центрируем диалог над master
        self.update_idletasks()
        mx = master.winfo_rootx()
        my = master.winfo_rooty()
        mw = master.winfo_width()
        mh = master.winfo_height()
        w = self.winfo_width()
        h = self.winfo_height()
        x = mx + (mw - w) // 2
        y = my + (mh - h) // 2
        self.geometry(f"+{x}+{y}")

    def _build_calendar(self):
        self.lbl_month.config(text=f"{calendar.month_name[self.month]} {self.year}")
        for w in self.frm_days.winfo_children():
            w.destroy()
        days = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
        for idx, d in enumerate(days):
            ttk.Label(self.frm_days, text=d).grid(row=0, column=idx, padx=2, pady=2)
        monthcal = calendar.monthcalendar(self.year, self.month)
        for r, week in enumerate(monthcal, start=1):
            for c, day in enumerate(week):
                if day == 0:
                    ttk.Label(self.frm_days, text=" ").grid(row=r, column=c, padx=2, pady=2)
                else:
                    btn = ttk.Button(
                        self.frm_days, text=f"{day:2d}", width=3,
                        command=lambda d=day: self._select_day(d)
                    )
                    btn.grid(row=r, column=c, padx=1, pady=1)

    def _prev_month(self):
        if self.month == 1:
            self.month = 12
            self.year -= 1
        else:
            self.month -= 1
        self._build_calendar()

    def _next_month(self):
        if self.month == 12:
            self.month = 1
            self.year += 1
        else:
            self.month += 1
        self._build_calendar()

    def _select_day(self, day: int):
        date_str = f"{self.year:04d}-{self.month:02d}-{day:02d}"
        self.entry.delete(0, "end")
        self.entry.insert(0, date_str)
        self.destroy()


def run_async(fn):
    def wrapper(*args, **kwargs):
        threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()
    return wrapper


class App(tb.Window):
    def __init__(self):
        super().__init__(themename="litera")
        self.title("Модель инвестиций")
        self.geometry("900x620")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Вкладки
        self.nb = ttk.Notebook(self)
        self.nb.grid(row=0, column=0, sticky="nsew")

        # Строка состояния
        self.status_bar = ttk.Label(self, text="Готово", relief="sunken", anchor="w")
        self.status_bar.grid(row=1, column=0, sticky="ew")

        # Хранилища данных
        self.raw_df: pd.DataFrame | None = None
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.mm: ModelManager | None = None
        self.y_pred: np.ndarray | None = None
        self.weights: pd.Series | None = None

        self._build_tab_data()
        self._build_tab_portfolio()
        self._build_tab_model()
        self._add_tooltips_and_validation()

    def _build_tab_data(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Данные")
        for c in range(4):
            tab.columnconfigure(c, weight=1)

        # С (дата)
        ttk.Label(tab, text="С (ГОД-МЕСЯЦ-ДЕНЬ)").grid(
            row=0, column=0, sticky="w", padx=8, pady=6
        )
        self.ent_from = ttk.Entry(tab)
        self.ent_from.insert(0, "2009-01-01")
        self.ent_from.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(tab, text="📅", width=2,
                   command=lambda: CalendarDialog(self, self.ent_from))\
            .grid(row=0, column=2, sticky="w")

        # По (дата)
        ttk.Label(tab, text="По").grid(
            row=1, column=0, sticky="w", padx=8, pady=6
        )
        self.ent_to = ttk.Entry(tab)
        self.ent_to.insert(0, "2022-01-31")
        self.ent_to.grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(tab, text="📅", width=2,
                   command=lambda: CalendarDialog(self, self.ent_to))\
            .grid(row=1, column=2, sticky="w")

        # Листинг
        ttk.Label(tab, text="Листинг").grid(row=0, column=3, sticky="w")
        self.combo_lvl = ttk.Combobox(
            tab, values=[1, 2, "1+2"], state="readonly", width=7
        )
        self.combo_lvl.set("1")
        self.combo_lvl.grid(row=0, column=3, sticky="e", padx=8)

        # Тикеры (сдвинуто под даты)
        ttk.Label(tab, text="Тикеры (через запятую)").grid(
            row=2, column=0, sticky="w", padx=8, pady=(12,0)
        )
        self.ent_tickers = ttk.Entry(tab)
        self.ent_tickers.grid(row=2, column=1, columnspan=3, sticky="ew", padx=8, pady=(12,0))

        # Прогрессбар и метка
        self.prog = ttk.Progressbar(tab, mode="indeterminate")
        self.prog.grid(row=3, columnspan=4, sticky="ew", padx=8, pady=15)

        # Кнопки по центру
        ttk.Button(tab, text="Собрать", command=self.collect_async)\
            .grid(row=5, column=1, padx=6, pady=10)
        ttk.Button(tab, text="Загрузить CSV", command=self.load_csv)\
            .grid(row=5, column=2, padx=6, pady=10)

    @run_async
    def collect_async(self):
        self.after(0, lambda: self.status_bar.config(text="Сбор данных…"))
        try:
            lvls = (1, 2) if self.combo_lvl.get() == "1+2" else (int(self.combo_lvl.get()),)
            custom = [s.strip().upper() for s in self.ent_tickers.get().split(",") if s.strip()]
            self.prog.start(12)
            dc = DataCollector(
                self.ent_from.get(), self.ent_to.get(),
                listing_levels=lvls, custom_secids=custom or None
            )
            self.raw_df = dc.collect()
            dc.save_csv("history.csv")
            self.after(0, lambda: self.status_bar.config(
                text=f"Собрано {len(self.raw_df)} строк → history.csv"
            ))
        except Exception as e:
            self.after(0, lambda: self.status_bar.config(text="Ошибка сбора данных"))
            messagebox.showerror("Ошибка сбора", str(e))
        finally:
            self.prog.stop()

    def load_csv(self):
        self.status_bar.config(text="Загрузка CSV…")
        path = filedialog.askopenfilename(
            title="Выбрать CSV-файл",
            filetypes=[("CSV", "*.csv;*.txt")]
        )
        if not path:
            self.status_bar.config(text="Готово")
            return
        try:
            df = pd.read_csv(
                path, sep=";", encoding="utf-8-sig",
                parse_dates=["TRADEDATE"]
            )
            for col in ("MARKETPRICE3", "CLOSE"):
                if col in df.columns:
                    df[col] = (
                        df[col].astype(str)
                            .str.replace('="', '', regex=False)
                            .str.replace('"',  '', regex=False)
                            .replace('', np.nan)
                            .astype(float)
                    )
            if "VOLUME" in df.columns:
                df["VOLUME"] = pd.to_numeric(df["VOLUME"], errors="coerce")
            self.raw_df = df
            fname = Path(path).name
            self.status_bar.config(text=f"Загружено {len(df)} строк из {fname}")
        except Exception as e:
            self.status_bar.config(text="Ошибка загрузки CSV")
            messagebox.showerror("Ошибка загрузки CSV", str(e))

    def _build_tab_portfolio(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Портфель")
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(1, weight=1)

        ttk.Label(tab, text="Сколько бумаг в портфеле").grid(
            row=0, column=0, sticky="w", padx=8)
        self.spin_topn = ttk.Spinbox(tab, from_=1, to=50, width=6)
        self.spin_topn.set(10)
        self.spin_topn.grid(row=0, column=1, sticky="w", padx=8)

        ttk.Button(tab, text="Прогноз + Оптимизировать", command=self.opt_async)\
            .grid(row=0, column=2, padx=8, pady=10)

        self.tree = ttk.Treeview(tab, columns=("weight",), show="headings", height=14)
        self.tree.heading("weight", text="ТИКЕР : соотношение в процентах")
        self.tree.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=8)

        self.fig = mpf.Figure(figsize=(4, 3))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab)
        self.canvas.get_tk_widget().grid(
            row=1, column=2, sticky="nsew", padx=8, pady=8
        )
        toolbar_frame = ttk.Frame(tab)
        toolbar_frame.grid(row=2, column=2, sticky="ew", padx=8)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

    @run_async
    def opt_async(self):
        self.after(0, lambda: self.status_bar.config(text="Оптимизация портфеля…"))
        if self.mm is None:
            messagebox.showwarning("Нет модели", "Сначала загрузите или обучите модель")
            self.after(0, lambda: self.status_bar.config(text="Готово"))
            return
        if self.X is None or self.y is None:
            if self.raw_df is None:
                messagebox.showwarning("Нет данных", "Сначала соберите или загрузите данные")
                self.after(0, lambda: self.status_bar.config(text="Готово"))
                return
            n_lags = int(self.spin_lags.get())
            ma_tuple = tuple(int(x.strip()) for x in self.ent_ma.get().split(",") if x.strip())
            proc = DataProcessor(n_lags=n_lags, ma_windows=ma_tuple, vol_windows=(ma_tuple[0],))
            self.X, self.y = proc.transform(self.raw_df)

        try:
            topn = int(self.spin_topn.get())
            self.y_pred = self.mm.predict(self.X)
            opt = Optimizer(self.X, self.y_pred, self.raw_df, lookback=252, top_n=topn)
            self.weights = opt.compute()
            opt.save_report(self.weights, "portfolio_report.xlsx")

            self.tree.delete(*self.tree.get_children())
            for secid, w in self.weights.items():
                self.tree.insert("", "end", values=(f"{secid}: {w:.3%}",))

            self.ax.clear()
            self.ax.bar(self.weights.index, self.weights.values)
            self.ax.set_title(f"Распределение {topn} тикеров на графике")
            self.ax.tick_params(axis="x", rotation=45)
            self.fig.tight_layout()
            self.canvas.draw_idle()

            self.after(0, lambda: self.status_bar.config(
                text="Портфель оптимизирован → portfolio_report.xlsx"
            ))
            messagebox.showinfo("Готово", "portfolio_report.xlsx сохранён")
        except Exception as e:
            self.after(0, lambda: self.status_bar.config(text="Ошибка оптимизации"))
            messagebox.showerror("Ошибка оптимизации", str(e))

    def _build_tab_model(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="Модель")

        frm_btns = ttk.Frame(tab)
        frm_btns.pack(pady=12)
        ttk.Button(frm_btns, text="Загрузить модель", command=self.load_model).grid(row=0, column=0, padx=8)
        ttk.Button(frm_btns, text="Обучить новую", command=self.train_async).grid(row=0, column=1, padx=8)

        frm_opt = ttk.LabelFrame(tab, text="Параметры признаков", padding=8)
        frm_opt.pack(fill="x", padx=8)

        ttk.Label(frm_opt, text="n_lags").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        self.spin_lags = ttk.Spinbox(frm_opt, from_=1, to=20, width=5)
        self.spin_lags.set(12)
        self.spin_lags.grid(row=0, column=1, pady=4)

        ttk.Label(frm_opt, text="MA окна (через запятую)").grid(row=0, column=2, sticky="w")
        self.ent_ma = ttk.Entry(frm_opt, width=18)
        self.ent_ma.insert(0, "5,10,20,60")
        self.ent_ma.grid(row=0, column=3, padx=4, pady=4)

        self.lbl_metrics = ttk.Label(tab, text="—")
        self.lbl_metrics.pack(pady=10)

    def load_model(self):
        path = filedialog.askopenfilename(
            title="Выбрать model.pkl",
            filetypes=[("Pickle-model", "*.pkl")],
            initialfile="model.pkl"
        )
        if not path:
            return
        try:
            self.mm = ModelManager.load(path)
            text = f"Модель загружена · R²={self.mm.r2_:.3f}  RMSE={self.mm.rmse_:.4f}"
            self.lbl_metrics.config(text=text)
            self.status_bar.config(text=text)
        except Exception as e:
            self.status_bar.config(text="Ошибка загрузки модели")
            messagebox.showerror("Ошибка загрузки", str(e))

    @run_async
    def train_async(self):
        self.after(0, lambda: self.status_bar.config(text="Обучение модели…"))
        if self.raw_df is None:
            messagebox.showwarning("Нет данных", "Сначала соберите или загрузите данные")
            self.after(0, lambda: self.status_bar.config(text="Готово"))
            return
        try:
            n_lags = int(self.spin_lags.get())
            ma_tuple = tuple(int(x.strip()) for x in self.ent_ma.get().split(",") if x.strip())
            proc = DataProcessor(n_lags=n_lags, ma_windows=ma_tuple, vol_windows=(ma_tuple[0],))
            self.X, self.y = proc.transform(self.raw_df)

            self.mm = ModelManager()
            self.mm.fit(self.X, self.y)
            self.mm.save("model.pkl")
            text = f"Модель обучена · R²={self.mm.r2_:.3f}  RMSE={self.mm.rmse_:.4f}"
            self.lbl_metrics.config(text=text)
            self.after(0, lambda: self.status_bar.config(text=text))
        except Exception as e:
            self.after(0, lambda: self.status_bar.config(text="Ошибка обучения модели"))
            messagebox.showerror("Ошибка обучения", str(e))

    def _add_tooltips_and_validation(self):
        ToolTip(self.ent_from,    "Введите или выберите дату начала (ГГГГ-ММ-ДД)")
        ToolTip(self.ent_to,      "Введите или выберите дату окончания")
        ToolTip(self.combo_lvl,   "Уровень листинга: 1, 2 или 1+2")
        ToolTip(self.ent_tickers, "Тикеры через запятую (например: AAPL, GOOG)")
        ToolTip(self.spin_topn,   "Сколько бумаг в портфеле (1–50)")
        ToolTip(self.spin_lags,   "Количество лагов признаков (1–20)")
        ToolTip(self.ent_ma,      "Окна скользящих средних через запятую (например: 5,10,20)")

        for entry in (self.ent_from, self.ent_to):
            entry.bind("<FocusOut>", self._validate_date)
        vcmd = (self.register(self._validate_int), '%P')
        for sb in (self.spin_lags, self.spin_topn):
            sb.config(validate='key', validatecommand=vcmd)

    def _validate_date(self, event):
        w = event.widget
        val = w.get()
        try:
            datetime.datetime.strptime(val, "%Y-%m-%d")
            w.config(foreground="black")
            self.status_bar.config(text="Готово")
        except ValueError:
            w.config(foreground="red")
            self.status_bar.config(text=f"Некорректная дата: {val}")

    def _validate_int(self, proposed: str):
        if proposed == "" or proposed.isdigit():
            return True
        self.bell()
        return False


if __name__ == "__main__":
    App().mainloop()
