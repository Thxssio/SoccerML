import threading
from stable_baselines3.common.callbacks import BaseCallback

class RenderCallback(BaseCallback):
    """
    Callback para renderizar o ambiente a cada render_freq passos.
    ATENÇÃO: Renderizar durante o treinamento pode diminuir a velocidade do treinamento.
    """
    def __init__(self, render_freq: int = 100, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            threading.Thread(target=self._render_env, daemon=True).start()
        return True

    def _render_env(self):
        """
        Renderiza o ambiente de forma assíncrona para não travar o treinamento.
        """
        try:
            self.training_env.envs[0].render()
        except Exception as e:
            print(f"Erro ao renderizar: {e}")
