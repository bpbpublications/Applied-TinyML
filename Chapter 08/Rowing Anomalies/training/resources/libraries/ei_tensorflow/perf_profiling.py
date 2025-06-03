from __future__ import print_function
import json
import traceback
import os
import json, time, traceback
import shutil, time, subprocess, math
from concurrent.futures import ThreadPoolExecutor

def ei_log(msg: str):
    print("EI_LOG_LEVEL=debug", msg, flush=True)

def run_tasks_in_parallel(tasks, parallel_count):
    res = []
    with ThreadPoolExecutor(parallel_count) as executor:
        running_tasks = [executor.submit(task) for task in tasks]
        for running_task in running_tasks:
            res.append(running_task.result())
    return res

def calculate_memory(model_file, model_type, prepare_model_tflite_script, prepare_model_tflite_eon_script,
                     calculate_non_cmsis=False, calculate_eon_ram_optimized=False):
    model_size = os.stat(model_file).st_size

    parallel_count = 5
    if (model_size > 1 * 1024 * 1024):
        parallel_count = 1

    # Some models don't have the scripts (e.g. akida) so skip this step
    if prepare_model_tflite_script or prepare_model_tflite_eon_script:
        memory = {}

        def calc_memory(id, title, is_eon, is_non_cmsis, is_eon_ram_optimized):
            try:
                print('Profiling ' + model_type + ' model (' + title + ')...', flush=True)

                benchmark_folder = f'/app/benchmark-{id}'
                script = f'{benchmark_folder}/prepare_tflite_{id}.sh'
                if (is_eon):
                    if (is_non_cmsis):
                        script = f'{benchmark_folder}/prepare_eon_cmsisnn_disabled_{id}.sh'
                    else:
                        script = f'{benchmark_folder}/prepare_eon_{id}.sh'

                out_folder = f'{benchmark_folder}/tflite-model'

                # create prep scripts
                if is_eon:
                    if is_non_cmsis:
                        with open(script, 'w') as f:
                            f.write(prepare_model_tflite_eon_script(model_file, cmsisnn=False, out_folder=out_folder, is_eon_ram_optimized=is_eon_ram_optimized))
                    else:
                        with open(script, 'w') as f:
                            f.write(prepare_model_tflite_eon_script(model_file, cmsisnn=True, out_folder=out_folder, is_eon_ram_optimized=is_eon_ram_optimized))
                else:
                    with open(script, 'w') as f:
                        f.write(prepare_model_tflite_script(model_file, out_folder=out_folder))

                args = [
                    f'{benchmark_folder}/benchmark.sh',
                    '--tflite-type', model_type,
                    '--tflite-file', model_file
                ]
                if is_eon:
                    args.append('--eon')
                if is_non_cmsis:
                    args.append('--disable-cmsis-nn')

                if os.path.exists(f'{benchmark_folder}/tflite-model'):
                    shutil.rmtree(f'{benchmark_folder}/tflite-model')
                subprocess.check_output(['sh', script]).decode("utf-8")
                tflite_output = json.loads(subprocess.check_output(args).decode("utf-8"))
                if  os.getenv('DEBUG_LOGS') == '1':
                    print(tflite_output['logLines'])

                if is_eon:
                    # eon is always correct in memory
                    return { 'id': id, 'output': tflite_output }
                else:
                    # add fudge factor since the target architecture is different
                    # (q: can this go since the changes in https://github.com/edgeimpulse/edgeimpulse/pull/6268)
                    old_arena_size = tflite_output['arenaSize']
                    if "anomaly" in model_file:
                        fudge_factor = 0.25
                    else:
                        fudge_factor = 0.2
                    extra_arena_size = int(math.floor((math.ceil(old_arena_size) * fudge_factor) + 1024))

                    tflite_output['ram'] = tflite_output['ram'] + extra_arena_size
                    tflite_output['arenaSize'] = tflite_output['arenaSize'] + extra_arena_size

                    return { 'id': id, 'output': tflite_output }
            except Exception as err:
                print('WARN: Failed to get memory (' + title + '): ', end='')
                print(err, flush=True)
                return { 'id': id, 'output': None }

        task_list = []

        if prepare_model_tflite_script:
            task_list.append(lambda: calc_memory(id=1, title='TensorFlow Lite Micro', is_eon=False, is_non_cmsis=False, is_eon_ram_optimized=False))
            if calculate_non_cmsis:
                task_list.append(lambda: calc_memory(id=2, title='TensorFlow Lite Micro, HW optimizations disabled', is_eon=False, is_non_cmsis=True, is_eon_ram_optimized=False))
        if prepare_model_tflite_eon_script:
            task_list.append(lambda: calc_memory(id=3, title='EON', is_eon=True, is_non_cmsis=False, is_eon_ram_optimized=False))
            if calculate_non_cmsis:
                task_list.append(lambda: calc_memory(id=4, title='EON, HW optimizations disabled', is_eon=True, is_non_cmsis=True, is_eon_ram_optimized=False))
            if calculate_eon_ram_optimized:
                task_list.append(lambda: calc_memory(id=5, title='EON, RAM optimized', is_eon=True, is_non_cmsis=False, is_eon_ram_optimized=True))

        results = run_tasks_in_parallel(task_list, parallel_count)
        for r in results:
            if (r['id'] == 1):
                memory['tflite'] = r['output']
            elif (r['id'] == 2):
                memory['tflite_cmsis_nn_disabled'] = r['output']
            elif (r['id'] == 3):
                memory['eon'] = r['output']
            elif (r['id'] == 4):
                memory['eon_cmsis_nn_disabled'] = r['output']
            elif (r['id'] == 5):
                memory['eon_ram_optimized'] = r['output']

    else:
        memory = None

    return memory

def profile_tflite_file(file, model_type,
                        prepare_model_tflite_script,
                        prepare_model_tflite_eon_script,
                        calculate_inferencing_time,
                        calculate_is_supported_on_mcu,
                        calculate_non_cmsis,
                        patch_based_inference=False):
    metadata = {
        'tfliteFileSizeBytes': os.path.getsize(file)
    }

    if calculate_inferencing_time:
        try:
            args = '/app/profiler/build/profiling ' + file

            print('Calculating inferencing time...', flush=True)
            a = os.popen(args).read()
            metadata['performance'] = json.loads(a[a.index('{'):a.index('}')+1])
            print('Calculating inferencing time OK', flush=True)
        except Exception as err:
            print('Error while calculating inferencing time:', flush=True)
            print(err, flush=True)
            traceback.print_exc()
            metadata['performance'] = None
    else:
        metadata['performance'] = None

    if calculate_is_supported_on_mcu:
        is_supported_on_mcu, mcu_support_error = check_if_model_runs_on_mcu(file, log_messages=True)
        metadata['isSupportedOnMcu'] = is_supported_on_mcu
        metadata['mcuSupportError'] = mcu_support_error
    else:
        metadata['isSupportedOnMcu'] = True
        metadata['mcuSupportError'] = None

    if (metadata['isSupportedOnMcu']):
        metadata['memory'] = calculate_memory(file, model_type, prepare_model_tflite_script, prepare_model_tflite_eon_script,
                                              calculate_non_cmsis=calculate_non_cmsis, calculate_eon_ram_optimized=patch_based_inference)
    return metadata

def check_if_model_runs_on_mcu(file, log_messages):
    is_supported_on_mcu = True
    mcu_support_error = None

    try:
        if log_messages:
            print('Determining whether this model runs on MCU...')

        # first we'll do a quick check against full TFLite. If the arena size is >6MB, we don't even pass it through
        # EON (fixes issues like https://github.com/edgeimpulse/edgeimpulse/issues/8838)
        full_tflite_result = subprocess.run(['/app/tflite-find-arena-size/find-arena-size', file], stdout=subprocess.PIPE)
        if (full_tflite_result.returncode == 0):
            stdout = full_tflite_result.stdout.decode('utf-8')
            msg = json.loads(stdout)

            arena_size = msg['arena_size']
            # more than 6MB
            if arena_size > 6 * 1024 * 1024:
                is_supported_on_mcu = False
                mcu_support_error = 'Calculated arena size is >6MB'

        # exit early
        if not is_supported_on_mcu:
            return is_supported_on_mcu, mcu_support_error

        result = subprocess.run(['/app/eon_compiler/compiler', '--verify', file], stdout=subprocess.PIPE)
        if (result.returncode == 0):
            stdout = result.stdout.decode('utf-8')
            msg = json.loads(stdout)

            arena_size = msg['arena_size']
            # more than 6MB
            if arena_size > 6 * 1024 * 1024:
                is_supported_on_mcu = False
                mcu_support_error = 'Calculated arena size is >6MB'
            else:
                is_supported_on_mcu = True
        else:
            is_supported_on_mcu = False
            stdout = result.stdout.decode('utf-8')
            if stdout != '':
                mcu_support_error = stdout
            else:
                mcu_support_error = 'Verifying model failed with code ' + str(result.returncode) + ' and no error message'
        if log_messages:
            print('Determining whether this model runs on MCU OK')
    except Exception as err:
        print('Determining whether this model runs on MCU failed:', flush=True)
        print(err, flush=True)
        traceback.print_exc()
        is_supported_on_mcu = False
        mcu_support_error = str(err)

    return is_supported_on_mcu, mcu_support_error
