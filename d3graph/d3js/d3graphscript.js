
function d3graphscript(config = {
    // Default values
    width: 800,
    height: 600,
    charge: -250,
    distance: 0,
    directed: false,
    collision: 0.5
    }) {

  //Constants for the SVG
  var width = config.width;
  var height = config.height;

  //Set up the colour scale
  var color = d3.scale.category20();

  var force = d3.layout.force()
    .charge(config.charge)
    .linkDistance((d) => d.edge_distance || config.distance)
    //.linkDistance((d) => config.distance > 0 ? config.distance : d.edge_weight)
    .size([width, height]);

  // DRAGGING START
  function dragstarted(d) {
    d3.event.sourceEvent.stopPropagation();
    d3.select(this).classed("dragging", true);
  }

  function dragged(d) {
    d3.select(this).attr("cx", d.x = d3.event.x).attr("cy", d.y = d3.event.y);
  }

  function dragended(d) {
    d3.select(this).classed("dragging", false);
  }

  var drag = force.drag()
    .origin(function(d) { return d; })
    .on("dragstart", dragstarted)
    .on("drag", dragged)
    .on("dragend", dragended);

  // DRAGGING STOP

  //Append a SVG to the body of the html page. Assign this SVG as an object to svg
  var svg = d3.select("body").append("svg")
    .attr("width", width)
    .attr("height", height)
//    .call(d3.behavior.zoom().on("zoom", function () { svg.attr("transform", "translate(" + d3.event.translate + ")" + " scale(" + d3.event.scale + ")") }))
//    .on("dblclick.zoom", null)
    .append("g")

  //.on("dblclick", threshold); // EXPLODE ALL CONNECTED POINTS

  graphRec = JSON.parse(JSON.stringify(graph));

  //Creates the graph data structure out of the json data
  force.nodes(graph.nodes)
    .links(graph.links)
    .start();

 // Create all the line svgs but without locations yet
  var link = svg.selectAll(".link")
    .data(graph.links)
    .enter().append("line")
    .attr("class", "link")
    .attr('marker-start', function(d){ return 'url(#marker_' + d.marker_start + ')' })
	.attr("marker-end", function(d) {
    	if (config.directed) {return 'url(#marker_' + d.marker_end + ')' }})
    .style("stroke-width", function(d) {return d.edge_width;})          // LINK-WIDTH
    .style("stroke", function(d) {return d.color;})                     // EDGE-COLORS
//  .style("stroke-width", 1); // WIDTH OF THE LINKS
  ;

  // ADD TEXT ON THE EDGES (PART 1/2)
   var linkText = svg.selectAll(".link-text")
     .data(graph.links)
     .enter().append("text")
     .attr("class", "link-text")
     .attr("font-size", function(d) {return d.label_fontsize + "px";})
     .style("fill", function(d) {return d.label_color;})
     .style("font-family", "Arial")
     //.attr("transform", "rotate(90)")
     .text(function(d) { return d.label; });

  //Do the same with the circles for the nodes
  var node = svg.selectAll(".node")
    .data(graph.nodes)
    .enter().append("g")
    .attr("class", "node")
    .call(drag)
    .on('dblclick', connectedNodes); // HIGHLIGHT ON/OFF

  {{ CLICK_COMMENT }} node.on('click', color_on_click); // ON CLICK HANDLER


  node.append("circle")
    .attr("r", function(d) { return d.node_size; })					// NODE SIZE
    .style("fill", function(d) {return d.node_color;})				// NODE-COLOR
    .style("opacity", function(d) {return d.node_opacity;}) 	    // NODE-OPACITY
    .style("stroke-width", function(d) {return d.node_size_edge;})	// NODE-EDGE-SIZE
    .style("stroke", function(d) {return d.node_color_edge;})		// NODE-COLOR-EDGE
  //  .style("stroke", '#000')										// NODE-EDGE-COLOR (all black)

  // Text in nodes
  node.append("text")
    .attr("dx", 10)
    .attr("dy", ".35em")
    .text(function(d) {return d.node_name}) // NODE-TEXT
    .style("font-size", function(d) {return d.node_fontsize + "px";}) // set font size equal to node edge size
	.style("fill", function(d) {return d.node_fontcolor;}) // set the text fill color to the same as node color
	.style("font-family", "monospace");
//  .style("stroke", "gray");

  let showInHover = ["node_tooltip"]; // Tooltip
  node.append("title")
      .text((d) => Object.keys(d)
          .filter((key) => showInHover.indexOf(key) !== -1)
          .map((key) => `${d[key]}`)
          .join('\n')
      )

  //Now we are giving the SVGs co-ordinates - the force layout is generating the co-ordinates which this code is using to update the attributes of the SVG elements
  force.on("tick", function() {
    link.attr("x1", function(d) { return d.source.x; })
      .attr("y1", function(d) { return d.source.y; })
      .attr("x2", function(d) { return d.target.x; })
      .attr("y2", function(d) { return d.target.y; });
    d3.selectAll("circle").attr("cx", function(d) { return d.x; })
      .attr("cy", function(d) { return d.y; });
    d3.selectAll("text").attr("x", function(d) { return d.x; })
      .attr("y", function(d) { return d.y; })
	linkText.attr("x", function(d) { return (d.source.x + d.target.x) / 2; })  // ADD TEXT ON THE EDGES (PART 2/2)
      .attr("y", function(d) { return (d.source.y + d.target.y) / 2; })
      .attr("text-anchor", "middle");
      ;

    node.each(collide(config.collision)); //COLLISION DETECTION. High means a big fight to get untouchable nodes (default=0.5)

  });

  // --------- MARKER -----------

  var data_marker = [
    { id: 0, name: 'circle', path: 'M 0, 0  m -5, 0  a 5,5 0 1,0 10,0  a 5,5 0 1,0 -10,0', viewbox: '-6 -6 12 12' }
  , { id: 1, name: 'square', path: 'M 0,0 m -5,-5 L 5,-5 L 5,5 L -5,5 Z', viewbox: '-5 -5 10 10' }
  , { id: 2, name: 'arrow', path: 'M 0,0 m -5,-5 L 5,0 L -5,5 Z', viewbox: '-5 -5 10 10' }
  , { id: 3, name: 'stub', path: 'M 0,0 m -1,-5 L 1,-5 L 1,5 L -1,5 Z', viewbox: '-1 -5 2 10' }
  ]

  //console.log(JSON.stringify(link))

  svg.append("defs").selectAll("marker")
    .data(data_marker)
    .enter()
    .append('svg:marker')
      .attr('id', function(d){ return 'marker_' + d.name})
      .attr('markerHeight', 10)
      .attr('markerWidth', 10)
      //.attr('markerUnits', 'strokeWidth')
      .attr("markerUnits", "userSpaceOnUse")                   // Fix marker width
      .attr('orient', 'auto')
      //.attr('refX', -15)                                     // Offset marker-start
      .attr('refX', 15)                                        // Offset marker-end
      .attr('refY', 0)
      .attr('viewBox', function(d){ return d.viewbox })
      .append('svg:path')
	    //.attr("transform", "rotate(180)")                    // Marker-start mirrored
        .attr('d', function(d){ return d.path })               // Marker type
        //.style("fill", function(d) {return d.marker_color;}) // Marker color
        .style("fill", '#808080')                              // Marker color
        .style("stroke", '#808080')                            // Marker edge-color
        .style("opacity", 0.95)                                // Marker opacity
        .style("stroke-width", 1);                             // Marker edge thickness

  // --------- END MARKER -----------


  // collision detection

  var padding = 1, // separation between circles
    radius = 8;

  function collide(alpha) {
    var quadtree = d3.geom.quadtree(graph.nodes);
    return function(d) {
      var rb = 2 * radius + padding,
        nx1 = d.x - rb,
        nx2 = d.x + rb,
        ny1 = d.y - rb,
        ny2 = d.y + rb;
      quadtree.visit(function(quad, x1, y1, x2, y2) {
        if (quad.point && (quad.point !== d)) {
          var x = d.x - quad.point.x,
            y = d.y - quad.point.y,
            l = Math.sqrt(x * x + y * y);
          if (l < rb) {
            l = (l - rb) / l * alpha;
            d.x -= x *= l;
            d.y -= y *= l;
            quad.point.x += x;
            quad.point.y += y;
          }
        }
        return x1 > nx2 || x2 < nx1 || y1 > ny2 || y2 < ny1;
      });
    };
  }
  // collision detection end


  //Toggle stores whether the highlighting is on **********************
  var toggle = 0;
  //Create an array logging what is connected to what
  var linkedByIndex = {};
  for (i = 0; i < graph.nodes.length; i++) {
    linkedByIndex[i + "," + i] = 1;
  };
  graph.links.forEach(function(d) {
    linkedByIndex[d.source.index + "," + d.target.index] = 1;
  });
  //This function looks up whether a pair are neighbours
  function neighboring(a, b) {
    return linkedByIndex[a.index + "," + b.index];
  }


    // COLOR ON CLICK
	function color_on_click() {
		// Give the original color back
		d3.selectAll(".node")
		.select("circle")
		.style("fill", function(d) {return d.node_color;})
		.style("opacity", function(d) {return d.node_opacity;})
		.style("stroke", function(d) {return d.node_color_edge;})
		.style("stroke-width", function(d) {return d.edge_width;})
		.attr("r", function(d) { return d.node_size; })
		;

		// Set the color on click
		d3.select(this).select("circle")
		.style("fill", {{ CLICK_FILL }})
		.style("stroke", "{{ CLICK_STROKE }}")
		.style("stroke-width", {{ CLICK_STROKEW }})
		.attr("r", function(d) { return d.node_size*{{ CLICK_SIZE }}; })
		;}



  function connectedNodes() {
    if (toggle == 0) {
      //Reduce the opacity of all but the neighbouring nodes
      d = d3.select(this).node().__data__;
      node.style("opacity", function(o) {
        return neighboring(d, o) | neighboring(o, d) ? 1 : 0.1;
      });
      link.style("opacity", function(o) {
        return d.index == o.source.index | d.index == o.target.index ? 1 : 0.1;
      });
      //Reduce the op
      toggle = 1;
    } else {
      //Put them back to opacity=1
      node.style("opacity", 0.95);
      link.style("opacity", 1);

      toggle = 0;
    }
  }
  //*************************************************************


  //adjust threshold
  function threshold() {
    let thresh = this.value;

    // console.log('Setting threshold', thresh)
    graph.links.splice(0, graph.links.length);

    for (var i = 0; i < graphRec.links.length; i++) {
      if (graphRec.links[i].edge_weight > thresh) {
        graph.links.push(graphRec.links[i]);
      }
    }
    restart();
  }

  d3.select("#thresholdSlider").on("change", threshold);

  //Restart the visualisation after any node and link changes
  function restart() {

    link = link.data(graph.links);
    link.exit().remove();
    link.enter().insert("line", ".node").attr("class", "link");
    link.style("stroke-width", function(d) {return d.edge_width;});           // LINK-WIDTH AFTER BREAKING WITH SLIDER
    //link.style('marker-start', function(d){ return 'url(#marker_' + d.marker_start  + ')' })
	link.style("marker-end", function(d) {                                    // Include the markers.
		if (config.directed) {return 'url(#marker_' + d.marker_end + ')' }})
    link.style("stroke", function(d) {return d.color;});                      // EDGE-COLOR AFTER BREAKING WITH SLIDER

    node = node.data(graph.nodes);
    node.enter().insert("circle", ".cursor").attr("class", "node").attr("r", 5).call(force.drag);
    force.start();
  }
}
