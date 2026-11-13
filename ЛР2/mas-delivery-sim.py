import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

class RoverState(Enum):
    IDLE = "СВОБОДЕН"
    TO_RESTAURANT = "К РЕСТОРАНУ"
    DELIVERING = "ДОСТАВКА"
    TO_STATION = "К СТАНЦИИ"
    CHARGING = "ЗАРЯДКА"
    DEAD = "АВАРИЯ (0%)"


class Performative(Enum):
    CFP = "CFP"
    PROPOSE = "PROPOSE"
    REFUSE = "REFUSE"
    ACCEPT_PROPOSAL = "ACCEPT_PROPOSAL"
    REJECT_PROPOSAL = "REJECT_PROPOSAL"
    DIRECT_ASSIGN = "DIRECT_ASSIGN"
    REQUEST_SLOT = "REQUEST_SLOT"
    CONFIRM_SLOT = "CONFIRM_SLOT"


@dataclass
class Message:
    step: int
    performative: Performative
    sender: str
    receiver: str
    content: dict


@dataclass
class Order:
    order_id: int
    restaurant_pos: Tuple[float, float]
    client_pos: Tuple[float, float]
    created_step: int
    deadline_step: int
    weight: float
    assigned_rover: Optional[int] = None
    picked_up: bool = False
    delivered_step: Optional[int] = None

    @property
    def is_completed(self) -> bool:
        return self.delivered_step is not None


@dataclass
class Station:
    station_id: int
    pos: Tuple[float, float]
    capacity: int
    occupied_by: List[int] = field(default_factory=list)

    @property
    def free_slots(self) -> int:
        return max(0, self.capacity - len(self.occupied_by))


@dataclass
class RoverAgent:
    rover_id: int
    pos: Tuple[float, float]
    battery: float = 100.0
    speed: float = 2.0
    state: RoverState = RoverState.IDLE
    current_order: Optional[Order] = None
    target_station: Optional[Station] = None
    total_energy_spent: float = 0.0
    completed_orders: int = 0

    def distance_to(self, target: Tuple[float, float]) -> float:
        return math.hypot(self.pos[0] - target[0], self.pos[1] - target[1])

    def estimate_energy_needed(self, order: Order, nearest_station_dist: float) -> float:
        d1 = self.distance_to(order.restaurant_pos)
        d2 = math.hypot(
            order.restaurant_pos[0] - order.client_pos[0],
            order.restaurant_pos[1] - order.client_pos[1],
        )
        # Расход: пустой ход + ход с грузом + запас до зарядки
        e_to_rest = d1 * 0.35
        e_to_client = d2 * (0.35 + 0.12 * order.weight)
        e_safety = nearest_station_dist * 0.35 + 8.0
        return e_to_rest + e_to_client + e_safety

    def evaluate_cfp(self, order: Order, nearest_station_dist: float) -> Optional[Tuple[float, float]]:
        """Локальная функция оценки заявки агентом в протоколе Contract Net."""
        if self.state != RoverState.IDLE:
            return None
        req_energy = self.estimate_energy_needed(order, nearest_station_dist)
        if self.battery < req_energy:
            return None

        d_total = self.distance_to(order.restaurant_pos) + math.hypot(
            order.restaurant_pos[0] - order.client_pos[0],
            order.restaurant_pos[1] - order.client_pos[1],
        )
        eta = d_total / self.speed
        # Стоимость заявки учитывает время и штраф за низкий заряд батареи
        battery_penalty = (100.0 - self.battery) * 0.15
        bid_cost = eta + battery_penalty
        return bid_cost, eta

def hungarian_algorithm(cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Классический Венгерский алгоритм (O(N^3)) для решения задачи о назначениях.
    Реализован явно без вызова сторонних солверов.
    Принимает квадратную матрицу стоимостей размером N x N.
    Возвращает список пар (индекс_строки, индекс_столбца).
    """
    n = len(cost_matrix)
    if n == 0:
        return []

    # Потенциалы строк u, столбцов v, паросочетание p и пути way
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0

            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        # Восстановление увеличивающей цепи
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignments = []
    for j in range(1, n + 1):
        if p[j] > 0:
            assignments.append((p[j] - 1, j - 1))
    return assignments

class DeliverySimulation:
    def __init__(
        self,
        num_rovers: int = 12,
        num_stations: int = 3,
        station_capacity: int = 2,
        map_size: float = 100.0,
        mechanism: str = "contract_net",
        seed: int = 42,
    ):
        self.rng = random.Random(seed)
        self.map_size = map_size
        self.mechanism = mechanism  # "hungarian" или "contract_net"
        self.step_count = 0

        # Генерация ресторанов
        self.restaurants = [
            (self.rng.uniform(15, map_size - 15), self.rng.uniform(15, map_size - 15))
            for _ in range(5)
        ]

        # Генерация зарядных станций
        self.stations = [
            Station(
                station_id=i,
                pos=(self.rng.uniform(10, map_size - 10), self.rng.uniform(10, map_size - 10)),
                capacity=station_capacity,
            )
            for i in range(num_stations)
        ]

        # Инициализация флота роботов
        self.rovers = [
            RoverAgent(
                rover_id=i,
                pos=(self.rng.uniform(5, map_size - 5), self.rng.uniform(5, map_size - 5)),
                battery=self.rng.uniform(55.0, 100.0),
                speed=self.rng.uniform(2.2, 2.8),
            )
            for i in range(num_rovers)
        ]

        self.pending_orders: List[Order] = []
        self.active_orders: List[Order] = []
        self.completed_orders: List[Order] = []
        self.messages: List[Message] = []
        self.next_order_id = 1

    def log_message(self, perf: Performative, sender: str, receiver: str, content: dict):
        self.messages.append(
            Message(
                step=self.step_count,
                performative=perf,
                sender=sender,
                receiver=receiver,
                content=content,
            )
        )

    def spawn_orders(self, arrival_prob: float = 0.45):
        if self.rng.random() < arrival_prob:
            rest_pos = self.rng.choice(self.restaurants)
            client_pos = (
                self.rng.uniform(5, self.map_size - 5),
                self.rng.uniform(5, self.map_size - 5),
            )
            order = Order(
                order_id=self.next_order_id,
                restaurant_pos=rest_pos,
                client_pos=client_pos,
                created_step=self.step_count,
                deadline_step=self.step_count + 65,
                weight=round(self.rng.uniform(0.5, 3.0), 1),
            )
            self.next_order_id += 1
            self.pending_orders.append(order)

    def nearest_station(self, pos: Tuple[float, float]) -> Tuple[Station, float]:
        best_s = min(
            self.stations,
            key=lambda s: math.hypot(pos[0] - s.pos[0], pos[1] - s.pos[1]),
        )
        dist = math.hypot(pos[0] - best_s.pos[0], pos[1] - best_s.pos[1])
        return best_s, dist

    def distribute_centralized_hungarian(self):
        if not self.pending_orders:
            return

        idle_rovers = [r for r in self.rovers if r.state == RoverState.IDLE]
        if not idle_rovers:
            return

        # Берем батч заказов не больше числа свободных роботов
        batch_orders = self.pending_orders[: len(idle_rovers)]
        n = max(len(idle_rovers), len(batch_orders))
        BIG_COST = 1e6

        # Централизованный сбор телеметрии (1 запрос от каждого свободного робота)
        for r in idle_rovers:
            self.log_message(
                Performative.DIRECT_ASSIGN,
                f"Rover_{r.rover_id}",
                "CentralDispatcher",
                {"telemetry": r.battery},
            )

        # Построение квадратной матрицы стоимостей N x N
        matrix = [[BIG_COST] * n for _ in range(n)]
        for i, rover in enumerate(idle_rovers):
            _, st_dist = self.nearest_station(rover.pos)
            for j, order in enumerate(batch_orders):
                eval_res = rover.evaluate_cfp(order, st_dist)
                if eval_res is not None:
                    matrix[i][j] = eval_res[0]

        pairs = hungarian_algorithm(matrix)

        assigned_order_ids = set()
        for r_idx, o_idx in pairs:
            if r_idx < len(idle_rovers) and o_idx < len(batch_orders):
                if matrix[r_idx][o_idx] < BIG_COST:
                    rover = idle_rovers[r_idx]
                    order = batch_orders[o_idx]

                    rover.current_order = order
                    rover.state = RoverState.TO_RESTAURANT
                    order.assigned_rover = rover.rover_id
                    assigned_order_ids.add(order.order_id)
                    self.active_orders.append(order)

                    # Директивная команда от диспетчера
                    self.log_message(
                        Performative.DIRECT_ASSIGN,
                        "CentralDispatcher",
                        f"Rover_{rover.rover_id}",
                        {"order_id": order.order_id},
                    )

        self.pending_orders = [
            o for o in self.pending_orders if o.order_id not in assigned_order_ids
        ]

    def distribute_contract_net(self):
        if not self.pending_orders:
            return

        remaining_orders = []
        for order in self.pending_orders:
            proposals = []
            # Шаг 1: Рассылка CFP всем доступным роботам (широковещание)
            for rover in self.rovers:
                if rover.state in (RoverState.DEAD, RoverState.CHARGING):
                    continue
                self.log_message(
                    Performative.CFP,
                    "OrderBroker",
                    f"Rover_{rover.rover_id}",
                    {"order_id": order.order_id},
                )

                # Шаг 2: Локальная оценка и ответ (PROPOSE или REFUSE)
                _, st_dist = self.nearest_station(rover.pos)
                eval_res = rover.evaluate_cfp(order, st_dist)
                if eval_res is not None:
                    bid_cost, eta = eval_res
                    proposals.append((bid_cost, eta, rover))
                    self.log_message(
                        Performative.PROPOSE,
                        f"Rover_{rover.rover_id}",
                        "OrderBroker",
                        {"order_id": order.order_id, "cost": round(bid_cost, 2)},
                    )
                else:
                    self.log_message(
                        Performative.REFUSE,
                        f"Rover_{rover.rover_id}",
                        "OrderBroker",
                        {"order_id": order.order_id},
                    )

            # Шаг 3: Выбор победителя и рассылка ACCEPT / REJECT
            if proposals:
                proposals.sort(key=lambda x: x[0])
                winner_cost, _, winner_rover = proposals[0]

                winner_rover.current_order = order
                winner_rover.state = RoverState.TO_RESTAURANT
                order.assigned_rover = winner_rover.rover_id
                self.active_orders.append(order)

                self.log_message(
                    Performative.ACCEPT_PROPOSAL,
                    "OrderBroker",
                    f"Rover_{winner_rover.rover_id}",
                    {"order_id": order.order_id},
                )

                for _, _, loser_rover in proposals[1:]:
                    self.log_message(
                        Performative.REJECT_PROPOSAL,
                        "OrderBroker",
                        f"Rover_{loser_rover.rover_id}",
                        {"order_id": order.order_id},
                    )
            else:
                remaining_orders.append(order)

        self.pending_orders = remaining_orders

    def _move_towards(self, rover: RoverAgent, target: Tuple[float, float], load_weight: float = 0.0) -> bool:
        dist = rover.distance_to(target)
        if dist <= rover.speed:
            rover.pos = target
            energy_used = dist * (0.35 + 0.12 * load_weight)
            rover.battery -= energy_used
            rover.total_energy_spent += energy_used
            return True

        dx = (target[0] - rover.pos[0]) / dist
        dy = (target[1] - rover.pos[1]) / dist
        rover.pos = (rover.pos[0] + dx * rover.speed, rover.pos[1] + dy * rover.speed)
        energy_used = rover.speed * (0.35 + 0.12 * load_weight)
        rover.battery -= energy_used
        rover.total_energy_spent += energy_used
        return False

    def step(self):
        self.step_count += 1
        self.spawn_orders()

        # Выбор механизма распределения задач
        if self.mechanism == "hungarian":
            self.distribute_centralized_hungarian()
        else:
            self.distribute_contract_net()

        # Обновление состояния каждого робота
        for rover in self.rovers:
            if rover.state == RoverState.DEAD:
                continue

            # Фоновый расход бортовой электроники
            rover.battery -= 0.08
            rover.total_energy_spent += 0.08

            if rover.battery <= 0:
                rover.battery = 0.0
                rover.state = RoverState.DEAD
                continue

            if rover.state == RoverState.IDLE:
                # Проверка необходимости подзарядки
                if rover.battery < 32.0:
                    st, _ = self.nearest_station(rover.pos)
                    self.log_message(
                        Performative.REQUEST_SLOT,
                        f"Rover_{rover.rover_id}",
                        f"Station_{st.station_id}",
                        {"battery": round(rover.battery, 1)},
                    )
                    if st.free_slots > 0:
                        st.occupied_by.append(rover.rover_id)
                        rover.target_station = st
                        rover.state = RoverState.TO_STATION
                        self.log_message(
                            Performative.CONFIRM_SLOT,
                            f"Station_{st.station_id}",
                            f"Rover_{rover.rover_id}",
                            {"slot": "reserved"},
                        )

            elif rover.state == RoverState.TO_RESTAURANT and rover.current_order:
                arrived = self._move_towards(rover, rover.current_order.restaurant_pos, 0.0)
                if arrived:
                    rover.current_order.picked_up = True
                    rover.state = RoverState.DELIVERING

            elif rover.state == RoverState.DELIVERING and rover.current_order:
                arrived = self._move_towards(
                    rover,
                    rover.current_order.client_pos,
                    rover.current_order.weight,
                )
                if arrived:
                    rover.current_order.delivered_step = self.step_count
                    self.completed_orders.append(rover.current_order)
                    self.active_orders.remove(rover.current_order)
                    rover.current_order = None
                    rover.completed_orders += 1
                    rover.state = RoverState.IDLE

            elif rover.state == RoverState.TO_STATION and rover.target_station:
                arrived = self._move_towards(rover, rover.target_station.pos, 0.0)
                if arrived:
                    rover.state = RoverState.CHARGING

            elif rover.state == RoverState.CHARGING and rover.target_station:
                rover.battery = min(100.0, rover.battery + 8.5)
                if rover.battery >= 95.0:
                    if rover.rover_id in rover.target_station.occupied_by:
                        rover.target_station.occupied_by.remove(rover.rover_id)
                    rover.target_station = None
                    rover.state = RoverState.IDLE

    def get_metrics(self) -> dict:
        delivered = len(self.completed_orders)
        avg_delivery_time = (
            np.mean([o.delivered_step - o.created_step for o in self.completed_orders])
            if delivered > 0
            else 0.0
        )
        late_orders = sum(
            1 for o in self.completed_orders if o.delivered_step > o.deadline_step
        )
        total_energy = sum(r.total_energy_spent for r in self.rovers)
        dead_rovers = sum(1 for r in self.rovers if r.state == RoverState.DEAD)

        return {
            "Выполнено заказов": delivered,
            "В очереди / в пути": len(self.pending_orders) + len(self.active_orders),
            "Среднее время доставки (шаги)": round(float(avg_delivery_time), 2),
            "Просрочено заказов": late_orders,
            "Всего сообщений (трафик)": len(self.messages),
            "Сообщений на 1 заказ": round(len(self.messages) / max(1, delivered), 2),
            "Суммарный расход энергии": round(total_energy, 1),
            "Разрядившихся роботов": dead_rovers,
        }

st.set_page_config(page_title="МАС Доставка — ЛР №2", layout="wide")
st.title("Лабораторная работа №2: Исследование механизмов распределения задач в МАС")
st.markdown(
    "Сравнение **Централизованной диспетчеризации (Венгерский алгоритм)** и **Децентрализованного протокола контрактных сетей (Contract Net Protocol)**."
)

with st.sidebar:
    st.header("Параметры симуляции")
    num_rovers = st.slider("Количество роботов-курьеров", 5, 40, 14)
    num_stations = st.slider("Количество зарядных станций", 1, 6, 3)
    station_cap = st.slider("Слотов на одной станции", 1, 5, 2)
    sim_steps = st.slider("Длительность симуляции (шагов)", 50, 300, 150)
    seed_val = st.number_input("Seed генератора (для воспроизводимости)", value=42, step=1)

    mode = st.radio(
        "Режим демонстрации",
        [
            "Сравнительный эксперимент (Оба алгоритма)",
            "Пошаговая визуализация: Венгерский алгоритм",
            "Пошаговая визуализация: Contract Net (CNP)",
        ],
    )

if mode == "Сравнительный эксперимент (Оба алгоритма)":
    if st.button("Запустить сравнительный эксперимент", type="primary"):
        sim_hungarian = DeliverySimulation(
            num_rovers=num_rovers,
            num_stations=num_stations,
            station_capacity=station_cap,
            mechanism="hungarian",
            seed=int(seed_val),
        )
        sim_cnp = DeliverySimulation(
            num_rovers=num_rovers,
            num_stations=num_stations,
            station_capacity=station_cap,
            mechanism="contract_net",
            seed=int(seed_val),
        )

        msg_history = []
        for s in range(sim_steps):
            sim_hungarian.step()
            sim_cnp.step()
            msg_history.append(
                {
                    "Шаг": s + 1,
                    "Венгерский алгоритм (сообщения)": len(sim_hungarian.messages),
                    "Contract Net (сообщения)": len(sim_cnp.messages),
                }
            )

        m_h = sim_hungarian.get_metrics()
        m_c = sim_cnp.get_metrics()

        df_comp = pd.DataFrame(
            {
                "Метрика": list(m_h.keys()),
                "Централизованный (Венгерский метод)": list(m_h.values()),
                "Децентрализованный (Contract Net)": list(m_c.values()),
            }
        )

        st.subheader("Сводная таблица показателей эффективности")
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            df_msg = pd.DataFrame(msg_history)
            fig_msg = px.line(
                df_msg,
                x="Шаг",
                y=["Венгерский алгоритм (сообщения)", "Contract Net (сообщения)"],
                title="Рост коммуникационных издержек (число сообщений)",
            )
            st.plotly_chart(fig_msg, use_container_width=True)

        with col2:
            bar_df = pd.DataFrame(
                {
                    "Механизм": ["Венгерский алгоритм", "Contract Net"],
                    "Среднее время доставки": [
                        m_h["Среднее время доставки (шаги)"],
                        m_c["Среднее время доставки (шаги)"],
                    ],
                    "Сообщений на 1 заказ": [
                        m_h["Сообщений на 1 заказ"],
                        m_c["Сообщений на 1 заказ"],
                    ],
                }
            )
            fig_bar = px.bar(
                bar_df,
                x="Механизм",
                y=["Среднее время доставки", "Сообщений на 1 заказ"],
                barmode="group",
                title="Сравнение времени доставки и цены координации",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

else:
    mech_key = "hungarian" if "Венгерский" in mode else "contract_net"
    sim = DeliverySimulation(
        num_rovers=num_rovers,
        num_stations=num_stations,
        station_capacity=station_cap,
        mechanism=mech_key,
        seed=int(seed_val),
    )
    for _ in range(sim_steps):
        sim.step()

    st.subheader(f"Карта городской среды на шаге {sim_steps}")
    fig = go.Figure()

    # Рестораны
    fig.add_trace(
        go.Scatter(
            x=[r[0] for r in sim.restaurants],
            y=[r[1] for r in sim.restaurants],
            mode="markers+text",
            marker=dict(size=14, color="orange", symbol="square"),
            text=[f"Рест.{i}" for i in range(len(sim.restaurants))],
            textposition="top center",
            name="Рестораны",
        )
    )

    # Станции зарядки
    fig.add_trace(
        go.Scatter(
            x=[s.pos[0] for s in sim.stations],
            y=[s.pos[1] for s in sim.stations],
            mode="markers+text",
            marker=dict(size=16, color="green", symbol="diamond"),
            text=[f"АЗС-{s.station_id}" for s in sim.stations],
            textposition="top center",
            name="Зарядные станции",
        )
    )

    # Роботы
    fig.add_trace(
        go.Scatter(
            x=[r.pos[0] for r in sim.rovers],
            y=[r.pos[1] for r in sim.rovers],
            mode="markers+text",
            marker=dict(
                size=12,
                color=[r.battery for r in sim.rovers],
                colorscale="RdYlGn",
                cmin=0,
                cmax=100,
                showscale=True,
                colorbar=dict(
                    title="Заряд %",
                    x=1.02,
                    y=0.5,
                    len=0.85,
                ),
            ),
            text=[f"R{r.rover_id} ({int(r.battery)}%)" for r in sim.rovers],
            textposition="bottom center",
            name="Роботы-курьеры",
        )
    )

    fig.update_layout(
        xaxis=dict(range=[0, 100], title="Координата X"),
        yaxis=dict(range=[0, 100], title="Координата Y"),
        height=550,
        margin=dict(t=60, r=90),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    col_m, col_log = st.columns([1, 1.3])
    with col_m:
        st.subheader("Метрики прогона")
        st.json(sim.get_metrics())
    with col_log:
        st.subheader("Последние сообщения протокола FIPA ACL")
        recent_msgs = [
            {
                "Шаг": m.step,
                "Тип": m.performative.value,
                "Отправитель": m.sender,
                "Получатель": m.receiver,
                "Данные": str(m.content),
            }
            for m in sim.messages[-15:]
        ]
        st.dataframe(pd.DataFrame(recent_msgs), use_container_width=True, hide_index=True)
