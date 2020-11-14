import subprocess
import re
import os


def main():
    # def get_response():
    #     succeeded = False
    #     result = ""
    #     while not succeeded:
    #         line = gtp_stream.stdout.readline()
    #         if line[0] == "=":
    #             succeeded = True
    #             line = line.strip()
    #             result = re.sub("^= ?", "", line)
    #     return result
    #
    # cmd = ["gnugo", "--mode", "gtp"]
    # pipe = subprocess.PIPE
    # gtp_stream = subprocess.Popen(cmd, stdin=pipe, stdout=pipe)
    # print(gtp_stream)
    #
    # cmd = "boardsize 19\n"
    # print(gtp_stream.stdin.write(cmd.encode()))
    # cmd = "showboard\n"
    # print(gtp_stream.stdin.write(cmd.encode()))
    # print(get_response())

    # subprocess.run(["pip", "list"], shell=True, check=True)
    # print(subprocess.run(["gnugo"], shell=True, check=True))
    # print(subprocess.run(["quit"], shell=True, check=True))

    # cmd = ["gnugo", "--mode", "gtp"]
    # proc = subprocess.Popen(
    #     ["gnugo.exe", "--mode", "gtp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    # )
    # stdout, stderr = proc.communicate()
    # print(stdout)
    # print(stderr)
    # proc.stdin.write("boardsize 19\n".encode())
    # line = proc.stdout.readline()
    # print(line)

    # com = "cmd"
    # # os.system(com)
    # subprocess.Popen(com)

    # proc = subprocess.Popen("date", stdout=subprocess.PIPE, shell=True)
    # output, err = proc.communicate()
    # procstat = proc.wait()

    # print(output)
    # print(procstat)

    subprocess.Popen(["start", "./alarm.wav"], shell=True)


if __name__ == "__main__":
    main()
