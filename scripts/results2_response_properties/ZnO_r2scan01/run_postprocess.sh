echo "Running ddk"
bash ../3_ddk.jl

for el in $(seq 1 3); do
    echo "Running electric field perturbation #$el"
    bash ../4a_perturb_elfield.jl $el &
done
wait

for s in $(seq 1 3); do
    echo "Running strain perturbation #$s"
    bash ../4b_perturb_strain.jl $s &
done
wait
for s in $(seq 4 6); do
    echo "Running strain perturbation #$s"
    bash ../4b_perturb_strain.jl $s &
done
wait

for at in $(seq 1 4); do
    for dir in $(seq 1 3); do
        echo "Running atom perturbation: atom #$at in direction $dir"
        bash ../4c_perturb_atom.jl $at $dir &
    done
    wait
done

echo "All finished!"