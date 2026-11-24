import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import Slider, Choice

# ---------------------------------------------------------
# 1. UI Component: แผงแดชบอร์ดคอนโซล (Advanced Medical Console UI)
# ---------------------------------------------------------
class InfoDisplay(TextElement):
    def __init__(self):
        super().__init__()

    def render(self, model):
        if model is None or model.schedule is None:
            return ""
            
        red_count = model.n_normal_cancer
        orange_count = model.n_mutated_cancer
        tcell_count = model.n_tcells
        
        def get_badge(is_active, active_text="ON", inactive_text="OFF", active_color="#2ECC71"):
            if is_active:
                return f"<span style='background-color: {active_color}; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;'>{active_text}</span>"
            return f"<span style='background-color: #E74C3C; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;'>{inactive_text}</span>"

        immuno_badge = get_badge(model.immunotherapy_active)
        targeted_badge = get_badge(model.targeted_therapy_active)
        cart_badge = get_badge(model.cart_therapy_activated, active_text="SUPERCHARGED", active_color="#00D2FF")
        
        # ค้นหาสถานะแผนการรักษารวม (Global Strategy Mode)
        if model.immunotherapy_active and model.targeted_therapy_active:
            strategy_text = "<span style='color: #F1C40F; font-weight: bold;'>⚠️ COMBINATION THERAPY ACTIVE</span>"
        elif model.immunotherapy_active:
            strategy_text = "<span style='color: #2ECC71; font-weight: bold;'>🧪 IMMUNOTHERAPY ACTIVE</span>"
        elif model.targeted_therapy_active:
            strategy_text = "<span style='color: #3498DB; font-weight: bold;'>🎯 TARGETED THERAPY ACTIVE</span>"
        else:
            strategy_text = "<span style='color: #7F8C8D;'>❌ NO TREATMENT (CONTROL GROUP)</span>"
        
        html_content = f"""
        <div style="background-color: #1E1E24; color: #ECF0F1; border: 2px solid #34495E; padding: 20px; border-radius: 8px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; border-bottom: 2px solid #34495E; padding-bottom: 10px; margin-bottom: 15px; align-items: center;">
                <span style="font-size: 18px; font-weight: bold; color: #00D2FF;">🖥️ BIOMEDICAL AGENT COMMAND CONSOLE</span>
                <div style="font-size: 14px; background: #2C2C35; padding: 5px 12px; border-radius: 20px; border: 1px solid #555;">
                    Mode: {strategy_text}
                </div>
                <span style="font-size: 16px; font-weight: bold; color: #F1C40F;">⏱️ Step: {model.schedule.steps}</span>
            </div>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 250px; background-color: #2C2C35; padding: 12px; border-radius: 6px; border-left: 4px solid #1F77B4;">
                    <b style="color: #AEC7E8; font-size: 14px; display: block; margin-bottom: 8px;">🧫 CELLULAR POPULATION COUNTS</b>
                    <div style="margin: 5px 0; display: flex; justify-content: space-between;">
                        <span>🔴 Normal Cancer (Red):</span>
                        <span style="font-weight: bold; color: #FF4B4B; font-size: 16px;">{red_count}</span>
                    </div>
                    <div style="margin: 5px 0; display: flex; justify-content: space-between;">
                        <span>🟠 Mutated Cancer (Orange):</span>
                        <span style="font-weight: bold; color: #FF9900; font-size: 16px;">{orange_count} <small style="color: #7F8C8D; font-size:11px;">(Limit: 5)</small></span>
                    </div>
                    <div style="margin: 5px 0; display: flex; justify-content: space-between;">
                        <span>🔵 Active Immune T-Cells:</span>
                        <span style="font-weight: bold; color: #00D2FF; font-size: 16px;">{tcell_count}</span>
                    </div>
                </div>
                <div style="flex: 1; min-width: 250px; background-color: #2C2C35; padding: 12px; border-radius: 6px; border-left: 4px solid #2CA02C;">
                    <b style="color: #98FB98; font-size: 14px; display: block; margin-bottom: 8px;">🧬 CLINICAL TREATMENT STATUS</b>
                    <div style="margin: 8px 0; display: flex; justify-content: space-between; align-items: center;">
                        <span>🧪 Immunotherapy Component:</span>
                        {immuno_badge}
                    </div>
                    <div style="margin: 8px 0; display: flex; justify-content: space-between; align-items: center;">
                        <span>🎯 Targeted Therapy Component:</span>
                        {targeted_badge}
                    </div>
                    <div style="margin: 8px 0; display: flex; justify-content: space-between; align-items: center;">
                        <span>🚀 CAR-T Gene Link:</span>
                        {cart_badge}
                    </div>
                </div>
            </div>
            <div style="font-size: 11px; color: #7F8C8D; margin-top: 10px; text-align: right;">💾 Auto-saving active: simulation_data.csv</div>
        </div>
        """
        return html_content

# ---------------------------------------------------------
# 1b. Spatial index: packed NumPy mirror of live cancer cells
# ---------------------------------------------------------
class CancerIndex:
    """Keeps the positions of all live cancer cells in packed NumPy arrays.

    T-cells need "which cancer cell is closest to me?" every step. Answering that
    by scanning schedule.agents costs O(all agents) *per T-cell* (and Mesa copies
    the whole agent set on every access). Here it is one vectorized distance
    computation.

    Ties (several cancer cells at the same distance) are broken by their position in
    the scheduler's current activation order -- exactly what the old linear scan over
    schedule.agents picked -- so seeded simulation results are unchanged.
    RankedActivation refreshes those ranks once per step after Mesa shuffles.
    """

    def __init__(self, capacity=64):
        self.x = np.empty(capacity, dtype=np.int64)
        self.y = np.empty(capacity, dtype=np.int64)
        self.ids = np.empty(capacity, dtype=np.int64)
        self.rank = np.empty(capacity, dtype=np.int64)  # position in scheduler order
        self.mutated = np.empty(capacity, dtype=bool)
        self.agents = [None] * capacity
        self.slot = {}
        self.n = 0
        self.next_rank = 0  # agents added mid-step go to the end of the order

    def _grow(self):
        cap = len(self.agents) * 2
        for name in ("x", "y", "ids", "rank", "mutated"):
            old = getattr(self, name)
            new = np.empty(cap, dtype=old.dtype)
            new[: self.n] = old[: self.n]
            setattr(self, name, new)
        self.agents.extend([None] * (cap - len(self.agents)))

    def add(self, agent):
        if self.n == len(self.agents):
            self._grow()
        i = self.n
        self.x[i], self.y[i] = agent.pos
        self.ids[i] = agent.unique_id
        self.rank[i] = self.next_rank
        self.next_rank += 1
        self.mutated[i] = agent.is_mutated
        self.agents[i] = agent
        self.slot[agent.unique_id] = i
        self.n += 1

    def remove(self, agent):
        i = self.slot.pop(agent.unique_id)
        last = self.n - 1
        if i != last:  # swap the last entry into the freed slot
            self.x[i], self.y[i] = self.x[last], self.y[last]
            self.ids[i] = self.ids[last]
            self.rank[i] = self.rank[last]
            self.mutated[i] = self.mutated[last]
            self.agents[i] = self.agents[last]
            self.slot[int(self.ids[i])] = i
        self.agents[last] = None
        self.n -= 1

    def move(self, agent):
        i = self.slot[agent.unique_id]
        self.x[i], self.y[i] = agent.pos

    def nearest(self, pos):
        """Closest live cancer cell to pos (squared Euclidean), or None."""
        n = self.n
        if n == 0:
            return None
        dx = self.x[:n] - pos[0]
        dy = self.y[:n] - pos[1]
        d2 = dx * dx + dy * dy
        best = np.flatnonzero(d2 == d2.min())
        if len(best) > 1:
            best = best[[int(np.argmin(self.rank[best]))]]
        return self.agents[int(best[0])]

    def refresh_ranks(self, ordered_agents):
        """Re-read the scheduler's activation order (call right after the shuffle)."""
        slot = self.slot
        rank = self.rank
        r = 0
        for agent in ordered_agents:
            i = slot.get(agent.unique_id)
            if i is not None:
                rank[i] = r
            r += 1
        self.next_rank = r

    @property
    def n_mutated(self):
        return int(self.mutated[: self.n].sum())


class RankedActivation(RandomActivation):
    """RandomActivation that tells the CancerIndex the new order after each shuffle.

    Same RNG use and same activation order as Mesa 2.3.0's RandomActivation.
    """

    def do_each(self, method, shuffle=False):
        if shuffle:
            self._agents.shuffle(inplace=True)
            self.model.cancer_index.refresh_ranks(self._agents)
        self._agents.do(method)


# ---------------------------------------------------------
# 2. Agent: ผู้วางแผนการรักษาจำลอง (AI Doctor)
# ---------------------------------------------------------
class DirectorAgent(Agent):
    def __init__(self, unique_id, model, threshold=5):
        super().__init__(unique_id, model)
        self.threshold = threshold
        self.is_triggered = False

    def step(self):
        if self.model.schedule is None:
            return

        current_orange_count = self.model.n_mutated_cancer
        current_red_count = self.model.n_normal_cancer
        current_tcell_count = self.model.n_tcells

        total_cancer = current_red_count + current_orange_count
        # AI Trigger conditions
        should_activate = (current_orange_count >= self.threshold) or (total_cancer >= 40) or (current_tcell_count <= 3)

        if should_activate and not self.is_triggered:
            self.model.immunotherapy_active = True
            self.model.targeted_therapy_active = True
            self.model.cart_therapy_activated = True
            self.is_triggered = True
            
            print(f"\n[AI Director Proactive Alert] Step {self.model.schedule.steps}: Emergency Threshold Met!")
            print(f" -> Status - Red: {current_red_count}, Orange: {current_orange_count}, T-Cells Left: {current_tcell_count}")
            print(" -> Action: Deploying COMBINATION & CAR-T THERAPY Immediately!")

            if current_tcell_count <= 4:
                print(" -> Action: Immunological Emergency! Recruiting 3 Fresh CAR-T Cells into the TME.")
                for _ in range(3):
                    new_tcell = TCell(self.model.next_id(), self.model, self.model.initial_tcell_energy)
                    self.model.schedule.add(new_tcell)
                    rx = self.random.randrange(self.model.grid.width)
                    ry = self.random.randrange(self.model.grid.height)
                    self.model.grid.place_agent(new_tcell, (rx, ry))
                    self.model.n_tcells += 1

# ---------------------------------------------------------
# 3. Agent: เซลล์มะเร็ง (มีระบบกลายพันธุ์ดื้อยา)
# ---------------------------------------------------------
class CancerCell(Agent):
    def __init__(self, unique_id, model, is_mutated=False):
        super().__init__(unique_id, model)
        self.is_mutated = is_mutated

    def step(self):
        if self.pos is None:
            return

        x, y = self.pos
        if 0 <= x < self.model.grid.width and 0 <= y < self.model.grid.height:
            self.model.tgf_grid[x][y] += 8.0 if self.is_mutated else 4.0

        if self.is_mutated:
            proliferation_rate = self.model.base_proliferation
        else:
            if self.model.targeted_therapy_active:
                proliferation_rate = self.model.base_proliferation * 0.2
            else:
                proliferation_rate = self.model.base_proliferation

        if self.random.random() < proliferation_rate:
            possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            if possible_steps:
                new_pos = self.random.choice(possible_steps)
                if new_pos is not None and self.model.grid.is_cell_empty(new_pos):
                    will_mutate = self.random.random() < 0.10 if not self.is_mutated else True
                    new_cancer = CancerCell(self.model.next_id(), self.model, is_mutated=will_mutate)
                    self.model.grid.place_agent(new_cancer, new_pos)
                    self.model.schedule.add(new_cancer)
                    self.model.cancer_index.add(new_cancer)

        if self.random.random() < 0.20:
            possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
            if possible_steps:
                new_position = self.random.choice(possible_steps)
                if new_position is not None:
                    self.model.grid.move_agent(self, new_position)
                    self.model.cancer_index.move(self)

# ---------------------------------------------------------
# 4. Agent: เม็ดเลือดขาว (T-Cell -> CAR-T Cell Evolution)
# ---------------------------------------------------------
class TCell(Agent):
    def __init__(self, unique_id, model, initial_energy):
        super().__init__(unique_id, model)
        self.energy = initial_energy

    def step(self):
        if self.pos is None:
            return

        x, y = self.pos
        if 0 <= x < self.model.grid.width and 0 <= y < self.model.grid.height:
            is_suppressed = self.model.tgf_grid[x][y] > 5.0
        else:
            is_suppressed = False

        if self.model.cart_therapy_activated:
            is_suppressed = False

        if self.energy <= 0:
            self.model.grid.remove_agent(self)
            self.model.schedule.remove(self)
            self.model.n_tcells -= 1
            return

        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        if not possible_steps:
            return
        
        ifn_values = []
        for p in possible_steps:
            if p is not None and 0 <= p[0] < self.model.grid.width and 0 <= p[1] < self.model.grid.height:
                ifn_values.append(self.model.ifn_grid[p[0]][p[1]])
            else:
                ifn_values.append(0.0)
        
        if len(ifn_values) > 0 and max(ifn_values) > 1.0:
            best_step_index = np.argmax(ifn_values)
            new_position = possible_steps[best_step_index]
        else:
            closest_cancer = self.model.cancer_index.nearest(self.pos)
            if closest_cancer is not None:
                target_pos = closest_cancer.pos
                def step_distance(step_pos, target_pos):
                    if step_pos is None or target_pos is None: return float('inf')
                    return (step_pos[0] - target_pos[0])**2 + (step_pos[1] - target_pos[1])**2
                new_position = min(possible_steps, key=lambda p: step_distance(p, target_pos))
            else:
                new_position = self.random.choice(possible_steps)

        if new_position is not None:
            self.model.grid.move_agent(self, new_position)
        
        self.energy -= 2 if is_suppressed else 1

        if self.pos is not None:
            cellmates = self.model.grid.get_cell_list_contents([self.pos])
            for agent in cellmates:
                if isinstance(agent, CancerCell):
                    is_target_mutated = agent.is_mutated
                    self.model.grid.remove_agent(agent)
                    self.model.schedule.remove(agent)
                    self.model.cancer_index.remove(agent)
                    
                    cx, cy = self.pos
                    if 0 <= cx < self.model.grid.width and 0 <= cy < self.model.grid.height:
                        self.model.ifn_grid[cx][cy] += 10.0
                    
                    if self.model.immunotherapy_active:
                        base_attack_cost = 2
                    else:
                        base_attack_cost = 12 if is_suppressed else 6
                    
                    if is_target_mutated and not self.model.cart_therapy_activated:
                        self.energy -= (base_attack_cost * 2)
                    else:
                        self.energy -= base_attack_cost
                    break

# ---------------------------------------------------------
# 5. Model: แกนกลางระบบจำลอง
# ---------------------------------------------------------
class TissueModel(Model):
    def __init__(self, num_cancer=25, num_tcell=8, tcell_energy=60, 
                 base_proliferation=0.06, drug_type="None", seed=None,
                 grid_size=20, autosave_csv=True):
        # Mesa 2.x reads the `seed` kwarg in Model.__new__ and builds self.random
        # from it, so declaring it here makes runs reproducible: same seed ->
        # identical trajectory. seed=None keeps the old random behavior.
        super().__init__()
        self.num_cancer = num_cancer
        self.num_tcell = num_tcell
        self.base_proliferation = base_proliferation
        self.initial_tcell_energy = tcell_energy
        self.grid = MultiGrid(grid_size, grid_size, False)
        self.schedule = RankedActivation(self)

        # Population bookkeeping kept incrementally so nothing has to scan every agent.
        self.cancer_index = CancerIndex()
        self.n_tcells = 0
        # GUI mode rewrites simulation_data.csv every step. Headless/batch runs should
        # pass autosave_csv=False and call export_csv() once at the end.
        self.autosave_csv = autosave_csv

        self.ifn_grid = np.zeros((grid_size, grid_size))
        self.tgf_grid = np.zeros((grid_size, grid_size))
        
        self.immunotherapy_active = (drug_type in ["Immunotherapy", "Combination Therapy"])
        self.targeted_therapy_active = (drug_type in ["Targeted Therapy", "Combination Therapy"])
        self.cart_therapy_activated = False

        for i in range(self.num_cancer):
            cancer = CancerCell(self.next_id(), self, is_mutated=False)
            self.schedule.add(cancer)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(cancer, (x, y))
            self.cancer_index.add(cancer)

        for i in range(self.num_tcell):
            tcell = TCell(self.next_id(), self, tcell_energy)
            self.schedule.add(tcell)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(tcell, (x, y))
            self.n_tcells += 1

        if drug_type == "AI Director Adaptive":
            director = DirectorAgent(self.next_id(), self, threshold=5)
            self.schedule.add(director)

        self.datacollector = DataCollector(
            model_reporters={
                "Normal Cancer (Red)": lambda m: m.n_normal_cancer,
                "Mutated Cancer (Orange)": lambda m: m.n_mutated_cancer,
                "Active T-Cells": lambda m: m.n_tcells,
                "Total IFN-g (Signal)": lambda m: np.sum(m.ifn_grid),
                "Total TGF-b (Toxin)": lambda m: np.sum(m.tgf_grid),
                "Combo Therapy Active (Indicator)": lambda m: 40.0 if (m.immunotherapy_active and m.targeted_therapy_active) else 0.0
            }
        )

    # ---- live population counts (O(1), no agent scan) ----
    @property
    def n_mutated_cancer(self):
        return self.cancer_index.n_mutated

    @property
    def n_normal_cancer(self):
        return self.cancer_index.n - self.cancer_index.n_mutated

    def export_csv(self, path="simulation_data.csv"):
        """Write the collected telemetry to CSV (same format the GUI autosave uses)."""
        self.datacollector.get_model_vars_dataframe().to_csv(path, index_label="Step")

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        self.ifn_grid *= 0.8
        self.tgf_grid *= 0.8

        if self.autosave_csv:
            self.export_csv("simulation_data.csv")

# ---------------------------------------------------------
# 6. ส่วนแสดงผล UI
# ---------------------------------------------------------
def cell_portrayal(agent):
    if agent is None: return
    portrayal = {"Layer": 1}

    if isinstance(agent, CancerCell):
        portrayal["Shape"] = "circle"
        portrayal["Color"] = "#FF9900" if agent.is_mutated else "#FF4B4B"
        portrayal["Filled"] = "true"
        portrayal["r"] = 0.6
    elif isinstance(agent, TCell):
        portrayal["Shape"] = "circle"
        if agent.pos is not None:
            x, y = agent.pos
            if agent.model.cart_therapy_activated:
                portrayal["Color"] = "#00D2FF" 
            elif 0 <= x < agent.model.grid.width and 0 <= y < agent.model.grid.height and agent.model.tgf_grid[x][y] > 5.0:
                portrayal["Color"] = "#9467BD"
            elif agent.energy > 15:
                portrayal["Color"] = "#1F77B4"
            else:
                portrayal["Color"] = "#AEC7E8"
        portrayal["Filled"] = "true"
        portrayal["r"] = 0.4
    return portrayal

canvas_element = CanvasGrid(cell_portrayal, 20, 20, 500, 500)
info_element = InfoDisplay()

cell_chart = ChartModule(
    [{"Label": "Normal Cancer (Red)", "Color": "#FF4B4B"},
     {"Label": "Mutated Cancer (Orange)", "Color": "#FF9900"},
     {"Label": "Active T-Cells", "Color": "#1F77B4"}],
    data_collector_name='datacollector'
)

chem_chart = ChartModule(
    [{"Label": "Total IFN-g (Signal)", "Color": "#2CA02C"}, 
     {"Label": "Total TGF-b (Toxin)", "Color": "#9467BD"},
     {"Label": "Combo Therapy Active (Indicator)", "Color": "#333333"}],  
    data_collector_name='datacollector'
)

model_params = {
    "num_cancer": Slider("Initial Cancer Cells", 25, 5, 100, 5),
    "num_tcell": Slider("Initial T-Cells", 8, 2, 50, 1),
    "tcell_energy": Slider("T-Cell Endurance (Energy)", 60, 20, 200, 10),
    "base_proliferation": Slider("Cancer Growth Rate", 0.06, 0.01, 0.20, 0.01),
    "drug_type": Choice("Select Treatment Strategy", 
                        value="None", 
                        choices=["None", 
                                 "Immunotherapy", 
                                 "Targeted Therapy", 
                                 "Combination Therapy", 
                                 "AI Director Adaptive"])
}

server = ModularServer(TissueModel, [canvas_element, info_element, cell_chart, chem_chart], 
                       "Biomedical Agent Platform (Master Final Edition)", model_params)
server.port = 8521

if __name__ == "__main__":
    server.launch()