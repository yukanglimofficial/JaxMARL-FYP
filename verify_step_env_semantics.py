import sys
import inspect
import jax
import jax.numpy as jnp
from jaxmarl import make

def l1_obs_diff(obs_a, obs_b, agents):
    return float(jax.device_get(sum(jnp.sum(jnp.abs(obs_a[a] - obs_b[a])) for a in agents)))

def main():
    print("Python:", sys.version)
    print("Python exe:", sys.executable)
    print("JAX version:", jax.__version__)
    print("Devices:", jax.devices())

    # LOCKED baseline env config from project.txt
    env = make("MPE_simple_spread_v3", N=3, local_ratio=0.5, max_cycles=25, action_type="Discrete")

    print("\nEnv:", "MPE_simple_spread_v3")
    print("Agents:", env.agents)
    print("Num agents:", env.num_agents)

    print("\nSignatures (audit):")
    try:
        print("  env.step     :", inspect.signature(env.step))
    except Exception as e:
        print("  env.step     : (could not inspect)", e)

    has_step_env = hasattr(env, "step_env")
    print("  Has step_env?:", has_step_env)
    if has_step_env:
        try:
            print("  env.step_env :", inspect.signature(env.step_env))
        except Exception as e:
            print("  env.step_env : (could not inspect)", e)

    if not has_step_env:
        print("\nNO-GO: env.step_env does not exist.")
        print("Project fallback required: disable terminal-geometry CSR/min-distance metrics.")
        return

    # Reset
    key = jax.random.PRNGKey(0)
    key, key_reset = jax.random.split(key)
    obs, state = env.reset(key_reset)

    # Find terminal transition using step_env
    terminal_saved = None

    for t in range(60):  # max_cycles=25 so this is plenty
        key, key_step, key_act = jax.random.split(key, 3)
        key_as = jax.random.split(key_act, env.num_agents)
        actions = {agent: env.action_space(agent).sample(key_as[i]) for i, agent in enumerate(env.agents)}

        state_before = state
        obs_post, state_post, rew_post, dones_post, infos_post = env.step_env(key_step, state, actions)

        done_all = bool(jax.device_get(dones_post["__all__"]))
        if done_all:
            terminal_saved = (key_step, state_before, actions)
            print(f"\nTerminal transition found at t={t} using step_env (done_all=True).")
            break

        obs, state = obs_post, state_post

    if terminal_saved is None:
        print("\nNO-GO: Did not reach done_all=True within 60 step_env steps.")
        print("This suggests max_cycles not being honored or stepping not progressing.")
        return

    # Compare step_env vs step on EXACT same (key, state, actions)
    key_term, state_term, actions_term = terminal_saved

    obs_env, state_env, rew_env, dones_env, infos_env = env.step_env(key_term, state_term, actions_term)
    obs_step, state_step, rew_step, dones_step, infos_step = env.step(key_term, state_term, actions_term)

    done_env = bool(jax.device_get(dones_env["__all__"]))
    done_step = bool(jax.device_get(dones_step["__all__"]))

    print("\nTerminal transition comparison:")
    print("  step_env done_all:", done_env)
    print("  step     done_all:", done_step)

    diff = l1_obs_diff(obs_env, obs_step, env.agents)
    print("  L1(obs_step_env - obs_step):", diff)

    if done_env and diff > 1e-6:
        print("\nGO: step_env returns a different observation than step at terminal transition.")
        print("=> Consistent with step() doing autoreset / post-terminal handling.")
        print("=> Use step_env for terminal-correct geometry metrics.")
    elif done_env and diff <= 1e-6:
        print("\nGO (with note): step_env and step observations match at terminal transition.")
        print("=> Still: use step_env as the wrapper-independent terminal observation source.")
    else:
        print("\nNO-GO: step_env did not report done_all=True as expected at terminal transition.")
        print("=> Investigate semantics before trusting terminal geometry metrics.")

if __name__ == "__main__":
    main()
