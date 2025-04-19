from settings import EnvSettings

def run_main() -> None:
    env_config = EnvSettings.load()
    print(env_config)


if __name__ == "__main__":
    run_main()
